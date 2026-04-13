import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import faiss
import anthropic
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
from PIL import Image
import io
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
# -------------------------------------------------------
# Page configuration -- must be the first Streamlit call
# sets the browser tab title, icon, and layout width
# -------------------------------------------------------
st.set_page_config(
    page_title="SentinelIQ",
    page_icon="sentinel",
    layout="wide"
)

# -------------------------------------------------------
# Load all models once using cache_resource
# Without this decorator models reload on every interaction
# -------------------------------------------------------
@st.cache_resource
def load_models():
    with open("../api/models/fraud_model.pkl", "rb") as f:
        fraud_model = pickle.load(f)
    with open("../api/models/scaler_fraud.pkl", "rb") as f:
        scaler_fraud = pickle.load(f)
    with open("../api/models/anomaly_model.pkl", "rb") as f:
        anomaly_model = pickle.load(f)
    with open("../api/models/scaler_anomaly.pkl", "rb") as f:
        scaler_anomaly = pickle.load(f)
    with open("../api/knowledge_base/chunks.json", "r") as f:
        knowledge_base = json.load(f)

    faiss_index = faiss.read_index(
        "../api/knowledge_base/faiss_index.bin")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    return (fraud_model, scaler_fraud, anomaly_model,
            scaler_anomaly, knowledge_base,
            faiss_index, embedding_model)

# Load everything
(fraud_model, scaler_fraud, anomaly_model,
 scaler_anomaly, knowledge_base,
 faiss_index, embedding_model) = load_models()

# Load dataset for visualisations
@st.cache_resource
def load_data():
    df = pd.read_csv("../data/Sample - Superstore.csv",
                     parse_dates=["Order Date", "Ship Date"],
                     encoding="latin-1")
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["YearMonth"] = df["Order Date"].dt.to_period("M")
    df["Profit_Margin"] = (df["Profit"] / df["Sales"] * 100).round(2)
    return df

df = load_data()

# -------------------------------------------------------
# Helper functions for user data upload and analysis
# -------------------------------------------------------

def parse_uploaded_file(uploaded_file):
    """
    Detects file type and parses it into the correct format.
    Returns a tuple of (dataframe_or_none, text_or_none, image_or_none)
    Only one of the three will be non-None depending on file type.
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        # Try common encodings in order
        # latin-1 handles special characters that utf-8 cannot
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=encoding)
                return df, None, None
            except UnicodeDecodeError:
                continue
        # If all encodings fail return None
        return None, None, None

    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
        return df, None, None

    elif filename.endswith(".pdf"):
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return None, text, None

    elif filename.endswith((".png", ".jpg", ".jpeg")):
        image = Image.open(uploaded_file)
        return None, None, image

    else:
        return None, None, None


def generate_dataframe_summary(df):
    """
    Automatically generates a text summary of a dataframe.
    This becomes the knowledge base for RAG querying.
    """
    summary_parts = []

    # Basic shape
    summary_parts.append(
        f"Dataset contains {df.shape[0]} rows and "
        f"{df.shape[1]} columns."
    )

    # Column names and types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object"]).columns.tolist()
    date_cols = df.select_dtypes(
        include=["datetime"]).columns.tolist()

    summary_parts.append(
        f"Numeric columns: {', '.join(numeric_cols) if numeric_cols else 'None'}."
    )
    summary_parts.append(
        f"Categorical columns: "
        f"{', '.join(categorical_cols) if categorical_cols else 'None'}."
    )

    # Missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        missing_str = ", ".join(
            [f"{col} ({val} missing)"
             for col, val in missing_cols.items()])
        summary_parts.append(f"Missing values found in: {missing_str}.")
    else:
        summary_parts.append("No missing values detected.")

    # Numeric statistics
    if len(numeric_cols) > 0:
        stats = df[numeric_cols].describe().round(2)
        for col in numeric_cols[:5]:  # limit to first 5 numeric cols
            summary_parts.append(
                f"Column '{col}': mean={stats.loc['mean', col]}, "
                f"min={stats.loc['min', col]}, "
                f"max={stats.loc['max', col]}, "
                f"std={stats.loc['std', col]}."
            )

    # Categorical value counts
    for col in categorical_cols[:3]:  # limit to first 3 categorical cols
        top_values = df[col].value_counts().head(3)
        top_str = ", ".join(
            [f"{val} ({count})"
             for val, count in top_values.items()])
        summary_parts.append(
            f"Column '{col}' top values: {top_str}."
        )

    return " ".join(summary_parts)


def build_dynamic_knowledge_base(df, summary_text):
    """
    Builds a FAISS-ready knowledge base from an uploaded dataframe.
    Returns knowledge_base list and faiss index.
    """
    chunks = []

    # Chunk 1 -- overall summary
    chunks.append({
        "topic": "dataset_overview",
        "text": summary_text
    })

    # Chunk 2 -- numeric column statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        stats_text = "Numeric column statistics. "
        for col in numeric_cols[:8]:
            stats = df[col].describe()
            stats_text += (
                f"{col}: mean {round(stats['mean'], 2)}, "
                f"median {round(stats['50%'], 2)}, "
                f"max {round(stats['max'], 2)}, "
                f"min {round(stats['min'], 2)}. "
            )
        chunks.append({
            "topic": "numeric_statistics",
            "text": stats_text
        })

    # Chunk 3 -- categorical breakdown
    categorical_cols = df.select_dtypes(
        include=["object"]).columns.tolist()
    if categorical_cols:
        cat_text = "Categorical column breakdown. "
        for col in categorical_cols[:5]:
            top = df[col].value_counts().head(5)
            cat_text += (
                f"{col} has {df[col].nunique()} unique values. "
                f"Top values: "
                f"{', '.join([str(v) for v in top.index.tolist()])}. "
            )
        chunks.append({
            "topic": "categorical_breakdown",
            "text": cat_text
        })

    # Chunk 4 -- anomaly findings if we ran detection
    if "Anomaly_Type" in df.columns:
        risk_count = len(df[df["Anomaly_Type"] == "Risk"])
        opp_count = len(df[df["Anomaly_Type"] == "Opportunity"])
        normal_count = len(df[df["Anomaly_Type"] == "Normal"])
        chunks.append({
            "topic": "anomaly_findings",
            "text": (
                f"Anomaly detection results. "
                f"Total transactions: {len(df)}. "
                f"Normal transactions: {normal_count}. "
                f"Risk anomalies detected: {risk_count}. "
                f"Opportunity anomalies detected: {opp_count}. "
                f"Risk anomalies are transactions with unusual "
                f"financial patterns indicating potential losses. "
                f"Opportunity anomalies show unusually high performance."
            )
        })

    # Build FAISS index from chunks
    texts = [chunk["text"] for chunk in chunks]
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = emb_model.encode(texts).astype(np.float32)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return chunks, index

# Anthropic client for RAG
client = anthropic.Anthropic(api_key="Your_Anthropic_API_Key")

# -------------------------------------------------------
# Sidebar navigation
# st.sidebar puts content in the left panel
# st.radio creates a set of selectable options
# -------------------------------------------------------
st.sidebar.title("SentinelIQ")
st.sidebar.markdown("Autonomous Business Intelligence & Risk Detection")

page = st.sidebar.radio(
    "Navigate",
    ["Business Intelligence",
     "Anomaly Detection",
     "Fraud Detection",
     "SentinelIQ Query",
     "Analyse Your Data"]
)

# -------------------------------------------------------
# Page 1 -- Business Intelligence Dashboard
# -------------------------------------------------------
if page == "Business Intelligence":
    st.title("Business Intelligence Dashboard")
    st.markdown("Sales and profit analysis from 2014 to 2017")

    # -- Monthly trend section --
    st.subheader("Monthly Sales and Profit Trends")

    monthly = df.groupby("YearMonth")[["Sales", "Profit"]].sum()
    monthly.index = monthly.index.to_timestamp()
    monthly["Sales_Rolling"] = monthly["Sales"].rolling(window=3).mean()
    monthly["Profit_Rolling"] = monthly["Profit"].rolling(window=3).mean()

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(monthly.index, monthly["Sales"],
                 color="steelblue", alpha=0.4, linewidth=1)
    axes[0].plot(monthly.index, monthly["Sales_Rolling"],
                 color="steelblue", linewidth=2, label="3-Month Avg")
    axes[0].set_title("Monthly Sales")
    axes[0].set_ylabel("Sales")
    axes[0].legend()

    axes[1].plot(monthly.index, monthly["Profit"],
                 color="seagreen", alpha=0.4, linewidth=1)
    axes[1].plot(monthly.index, monthly["Profit_Rolling"],
                 color="seagreen", linewidth=2, label="3-Month Avg")
    axes[1].set_title("Monthly Profit")
    axes[1].set_ylabel("Profit")
    axes[1].legend()

    plt.tight_layout()

    # st.pyplot renders matplotlib figures in Streamlit
    st.pyplot(fig)

    # -- Category and Region metrics --
    st.subheader("Category and Regional Performance")

    col1, col2 = st.columns(2)

    category_metrics = df.groupby("Category")[["Sales", "Profit"]].sum()
    category_metrics["Profit_Margin"] = (
        category_metrics["Profit"] /
        category_metrics["Sales"] * 100).round(2)

    region_metrics = df.groupby("Region")[["Sales", "Profit"]].sum()
    region_metrics["Profit_Margin"] = (
        region_metrics["Profit"] /
        region_metrics["Sales"] * 100).round(2)

    with col1:
        st.markdown("**Profit Margin by Category**")
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        bars = ax1.bar(category_metrics.index,
                       category_metrics["Profit_Margin"],
                       color=["steelblue", "seagreen", "tomato"])
        for i, val in enumerate(category_metrics["Profit_Margin"]):
            ax1.text(i, val + 0.2, f"{val}%", ha="center", fontsize=9)
        ax1.set_ylabel("Profit Margin (%)")
        plt.tight_layout()
        st.pyplot(fig1)

    with col2:
        st.markdown("**Sales vs Profit by Region**")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        x = range(len(region_metrics.index))
        width = 0.35
        ax2.bar([i - width/2 for i in x],
                region_metrics["Sales"],
                width=width, label="Sales", color="steelblue", alpha=0.8)
        ax2.bar([i + width/2 for i in x],
                region_metrics["Profit"],
                width=width, label="Profit", color="seagreen", alpha=0.8)
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(region_metrics.index)
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)

    # -- Automated insights panel --
    st.subheader("Automated Insights")

    insights = []
    LOW_MARGIN_THRESHOLD = 5.0

    for category, row in category_metrics.iterrows():
        if row["Profit_Margin"] < LOW_MARGIN_THRESHOLD:
            insights.append(
                f"WARNING: {category} has a low profit margin of "
                f"{row['Profit_Margin']}% -- investigate discounting or costs."
            )

    weakest_region = region_metrics["Profit_Margin"].idxmin()
    weakest_margin = region_metrics.loc[
        weakest_region, "Profit_Margin"].round(2)
    insights.append(
        f"ALERT: {weakest_region} is the weakest performing region "
        f"with a profit margin of {weakest_margin}%."
    )

    strongest_region = region_metrics["Profit_Margin"].idxmax()
    strongest_margin = region_metrics.loc[
        strongest_region, "Profit_Margin"].round(2)
    insights.append(
        f"INSIGHT: {strongest_region} is the top performing region "
        f"with a profit margin of {strongest_margin}%."
    )

    for insight in insights:
        if insight.startswith("WARNING") or insight.startswith("ALERT"):
            st.warning(insight)
        else:
            st.success(insight)
            
# -------------------------------------------------------
# Page 2 -- Anomaly Detection
# -------------------------------------------------------
elif page == "Anomaly Detection":
    st.title("Anomaly Detection")
    st.markdown("Enter a business transaction to check for anomalies.")

    # -- Input sliders --
    st.subheader("Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        sales = st.number_input("Sales Amount",
                                min_value=0.0,
                                max_value=50000.0,
                                value=500.0,
                                step=10.0)
        quantity = st.number_input("Quantity",
                                   min_value=1,
                                   max_value=50,
                                   value=3,
                                   step=1)

    with col2:
        discount = st.slider("Discount",
                             min_value=0.0,
                             max_value=1.0,
                             value=0.2,
                             step=0.05)
        profit = st.number_input("Profit",
                                 min_value=-5000.0,
                                 max_value=10000.0,
                                 value=100.0,
                                 step=10.0)

    # -- Run detection on button click --
    if st.button("Check for Anomaly"):
        features = np.array([[sales, quantity, discount, profit]])
        features_scaled = scaler_anomaly.transform(features)

        prediction = anomaly_model.predict(features_scaled)[0]
        score = anomaly_model.decision_function(features_scaled)[0]

        if prediction == 1:
            anomaly_type = "Normal"
        elif profit > 0:
            anomaly_type = "Opportunity"
        else:
            anomaly_type = "Risk"

        # -- Display result --
        st.subheader("Detection Result")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Anomaly Type", anomaly_type)
        with col2:
            st.metric("Anomaly Score", round(float(score), 4))
        with col3:
            st.metric("Is Anomaly", "Yes" if prediction == -1 else "No")

        if anomaly_type == "Risk":
            st.error("Risk anomaly detected -- this transaction has "
                     "an unusual financial profile and may indicate "
                     "a loss-making pattern.")
        elif anomaly_type == "Opportunity":
            st.success("Opportunity anomaly detected -- this transaction "
                       "shows unusually high profit and may represent "
                       "a best practice worth replicating.")
        else:
            st.info("Transaction appears normal based on historical patterns.")

        # -- Visualise on scatter plot --
        st.subheader("Transaction in Context")

        X_viz = df[["Discount", "Profit"]].copy()
        X_scaled_viz = scaler_anomaly.transform(
            df[["Sales", "Quantity", "Discount", "Profit"]])
        all_predictions = anomaly_model.predict(X_scaled_viz)

        fig, ax = plt.subplots(figsize=(10, 5))

        normal_mask = all_predictions == 1
        anomaly_mask = all_predictions == -1

        ax.scatter(df.loc[normal_mask, "Discount"],
                   df.loc[normal_mask, "Profit"],
                   color="steelblue", alpha=0.2, s=10, label="Normal")
        ax.scatter(df.loc[anomaly_mask, "Discount"],
                   df.loc[anomaly_mask, "Profit"],
                   color="tomato", alpha=0.6, s=20, label="Anomaly")

        # Plot the input transaction as a large star
        ax.scatter(discount, profit,
                   color="gold", s=300, marker="*",
                   zorder=5, label="Your Transaction")

        ax.set_xlabel("Discount")
        ax.set_ylabel("Profit")
        ax.set_title("Your Transaction vs Historical Patterns")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

# -------------------------------------------------------
# Page 3 -- Fraud Detection
# -------------------------------------------------------
elif page == "Fraud Detection":
    st.title("Fraud Detection")
    st.markdown("Enter transaction details to check fraud probability.")

    st.info(
        "V1 to V28 are PCA-transformed features from the original "
        "credit card dataset. In a real system these would be computed "
        "automatically from raw transaction data."
    )

    # -- Load a sample transaction from the dataset for easy testing --
    st.subheader("Quick Test")
    st.markdown(
        "Select a sample transaction from the test dataset "
        "to auto-fill the fields."
    )

    @st.cache_resource
    def load_fraud_data():
        df_fraud = pd.read_csv("../data/creditcard.csv")
        return df_fraud

    df_fraud = load_fraud_data()

    # Let user pick a normal or fraud sample
    sample_type = st.radio(
        "Sample type",
        ["Normal Transaction", "Known Fraud Transaction"],
        horizontal=True
    )

    if sample_type == "Known Fraud Transaction":
        sample = df_fraud[df_fraud["Class"] == 1].iloc[0]
    else:
        sample = df_fraud[df_fraud["Class"] == 0].iloc[0]

    # -- Run prediction on selected sample --
    if st.button("Analyse Selected Sample"):
        features = sample.drop("Class").values.reshape(1, -1)

        # Scale Time and Amount -- columns 0 and 29
        features_scaled = features.copy()
        features_scaled[:, [0, 29]] = scaler_fraud.transform(
            features[:, [0, 29]])

        fraud_prob = fraud_model.predict_proba(features_scaled)[0][1]

        if fraud_prob >= 0.8:
            risk_level = "HIGH"
        elif fraud_prob >= 0.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # -- Display results --
        st.subheader("Detection Result")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Fraud Probability",
                      f"{round(fraud_prob * 100, 2)}%")
        with col2:
            st.metric("Risk Level", risk_level)
        with col3:
            st.metric("Verdict",
                      "FRAUD" if fraud_prob >= 0.5 else "LEGITIMATE")

        # -- Probability gauge bar --
        st.subheader("Fraud Probability Gauge")

        fig, ax = plt.subplots(figsize=(10, 2))

        # Background bar
        ax.barh(0, 1, color="lightgray", height=0.5)

        # Probability bar coloured by risk level
        bar_color = (
            "tomato" if fraud_prob >= 0.8
            else "orange" if fraud_prob >= 0.5
            else "seagreen"
        )
        ax.barh(0, fraud_prob, color=bar_color, height=0.5)

        # Threshold line at 0.5
        ax.axvline(x=0.5, color="black",
                   linewidth=2, linestyle="--", label="Decision threshold")

        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Fraud Probability")
        ax.set_title(
            f"Fraud Probability: {round(fraud_prob * 100, 2)}% "
            f"-- Risk Level: {risk_level}"
        )
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

        if fraud_prob >= 0.5:
            st.error(
                f"Transaction flagged as potential fraud with "
                f"{round(fraud_prob * 100, 2)}% probability. "
                f"Recommend immediate review."
            )
        else:
            st.success(
                f"Transaction appears legitimate with only "
                f"{round(fraud_prob * 100, 2)}% fraud probability."
            )

        # -- Show transaction details --
        st.subheader("Transaction Details")
        st.dataframe(
            pd.DataFrame(sample.drop("Class")).T,
            use_container_width=True
        )
        
        
# -------------------------------------------------------
# Page 4 -- SentinelIQ Query
# -------------------------------------------------------
elif page == "SentinelIQ Query":
    st.title("SentinelIQ Query")
    st.markdown(
        "Ask questions about your business data in plain English. "
        "Powered by RAG and Claude AI."
    )

    # Initialise session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_suggestion" not in st.session_state:
        st.session_state.pending_suggestion = None

    # Suggested questions shown only when no conversation yet
    if len(st.session_state.messages) == 0:
        st.subheader("Suggested Questions")
        suggestions = [
            "Which region is underperforming and why?",
            "Which category should we be most worried about?",
            "Are there any dangerous transactions I should know about?",
            "What is the overall profit margin of the business?",
            "Which region has the most opportunities?"
        ]
        for suggestion in suggestions:
            if st.button(suggestion):
                st.session_state.pending_suggestion = suggestion

    # Display existing conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("Sources used"):
                    for source in message["sources"]:
                        st.markdown(f"- {source}")

    # Chat input box
    question = st.chat_input("Ask a question about your business data...")

    # Determine what question to process
    if question:
        pending_question = question
    elif st.session_state.pending_suggestion:
        pending_question = st.session_state.pending_suggestion
        st.session_state.pending_suggestion = None
    else:
        pending_question = None

    # Process the question
    if pending_question:
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": pending_question
        })

        # Display user message
        with st.chat_message("user"):
            st.markdown(pending_question)

        # Run RAG pipeline and display answer
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                question_embedding = embedding_model.encode(
                    [pending_question])
                question_embedding = question_embedding.astype(np.float32)
                faiss.normalize_L2(question_embedding)

                D, I = faiss_index.search(question_embedding, 3)

                context = "\n\n".join([
                    knowledge_base[i]["text"] for i in I[0]
                ])
                sources = [knowledge_base[i]["topic"] for i in I[0]]

                prompt = f"""You are SentinelIQ, a business intelligence assistant.
Use only the following business data to answer the question.
Be specific and use actual numbers from the context.

Context:
{context}

Question: {pending_question}"""

                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )

                answer = response.content[0].text

            st.markdown(answer)

            with st.expander("Sources used"):
                for source in sources:
                    st.markdown(f"- {source}")

        # Add assistant response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

# -------------------------------------------------------
# Page 5 -- Analyse Your Data
# -------------------------------------------------------
elif page == "Analyse Your Data":
    st.title("Analyse Your Data")
    st.markdown(
        "Upload any business dataset and SentinelIQ will "
        "automatically analyse it and answer your questions."
    )

    # -- File uploader --
    uploaded_file = st.file_uploader(
        "Upload your data file",
        type=["csv", "xlsx", "xls", "pdf", "png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")

        # -- Parse the file --
        with st.spinner("Parsing file..."):
            df_user, text_user, image_user = parse_uploaded_file(
                uploaded_file)

        # -------------------------------------------------------
        # Handle IMAGE files -- send to Claude vision API
        # -------------------------------------------------------
        if image_user is not None:
            st.subheader("Chart Analysis")
            st.image(image_user, caption="Uploaded Chart", width=600)

            with st.spinner("Analysing chart with Claude vision..."):
                # Convert image to bytes for API
                img_bytes = io.BytesIO()
                image_user.save(img_bytes, format="PNG")
                img_bytes = img_bytes.getvalue()

                import base64
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")

                # Send to Claude with vision
                vision_response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1000,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": img_b64
                                }
                            },
                            {
                                "type": "text",
                                "text": (
                                    "You are SentinelIQ, a business "
                                    "intelligence assistant. Analyse this "
                                    "chart or dashboard image and provide: "
                                    "1. What type of chart this is "
                                    "2. What data it shows "
                                    "3. Key trends or patterns visible "
                                    "4. Any anomalies or concerns "
                                    "5. Business recommendations "
                                    "Be specific and use any numbers "
                                    "visible in the chart."
                                )
                            }
                        ]
                    }]
                )

            st.subheader("AI Chart Interpretation")
            st.markdown(vision_response.content[0].text)

        # -------------------------------------------------------
        # Handle PDF files -- extract text and query with Claude
        # -------------------------------------------------------
        elif text_user is not None:
            st.subheader("PDF Analysis")

            if len(text_user.strip()) == 0:
                st.warning(
                    "Could not extract text from this PDF. "
                    "It may be a scanned image-based PDF."
                )
            else:
                st.info(
                    f"Extracted {len(text_user)} characters from PDF.")

                with st.expander("View extracted text"):
                    st.text(text_user[:2000] + "..." 
                            if len(text_user) > 2000 else text_user)

                with st.spinner("Generating PDF summary..."):
                    pdf_response = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=1000,
                        messages=[{
                            "role": "user",
                            "content": (
                                f"You are SentinelIQ, a business "
                                f"intelligence assistant. Analyse this "
                                f"business report and provide: "
                                f"1. Executive summary "
                                f"2. Key metrics mentioned "
                                f"3. Notable risks or concerns "
                                f"4. Recommendations "
                                f"\n\nReport text:\n{text_user[:4000]}"
                            )
                        }]
                    )

                st.subheader("AI Report Summary")
                st.markdown(pdf_response.content[0].text)

                # Enable querying on PDF content
                st.subheader("Ask Questions About This Report")
                pdf_question = st.text_input(
                    "Ask a question about the report...")

                if pdf_question:
                    with st.spinner("Searching report..."):
                        pdf_query_response = client.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=500,
                            messages=[{
                                "role": "user",
                                "content": (
                                    f"Based on this report, answer the "
                                    f"following question. Be specific "
                                    f"and use numbers where available."
                                    f"\n\nReport:\n{text_user[:4000]}"
                                    f"\n\nQuestion: {pdf_question}"
                                )
                            }]
                        )
                    st.markdown(pdf_query_response.content[0].text)

        # -------------------------------------------------------
        # Handle CSV and Excel files -- full analysis pipeline
        # -------------------------------------------------------
        elif df_user is not None:

            # -- Auto-generated summary --
            st.subheader("Dataset Overview")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df_user.shape[0])
            with col2:
                st.metric("Columns", df_user.shape[1])
            with col3:
                missing_total = df_user.isnull().sum().sum()
                st.metric("Missing Values", missing_total)

            st.dataframe(df_user.head(10), use_container_width=True)

            # -- Generate summary text --
            with st.spinner("Generating analysis..."):
                summary_text = generate_dataframe_summary(df_user)

            with st.expander("Auto-generated Data Summary"):
                st.write(summary_text)

            # -- Auto charts --
            st.subheader("Automatic Charts")

            numeric_cols = df_user.select_dtypes(
                include=[np.number]).columns.tolist()
            categorical_cols = df_user.select_dtypes(
                include=["object"]).columns.tolist()

            if len(numeric_cols) >= 2:
                # Distribution plots for first 4 numeric columns
                st.markdown("**Numeric Column Distributions**")
                cols_to_plot = numeric_cols[:4]
                fig, axes = plt.subplots(
                    1, len(cols_to_plot),
                    figsize=(14, 4))

                if len(cols_to_plot) == 1:
                    axes = [axes]

                for ax, col in zip(axes, cols_to_plot):
                    ax.hist(df_user[col].dropna(),
                            bins=30, color="steelblue", alpha=0.7)
                    ax.set_title(col)
                    ax.set_ylabel("Count")

                plt.tight_layout()
                st.pyplot(fig)

                # Correlation heatmap
                if len(numeric_cols) >= 3:
                    st.markdown("**Correlation Heatmap**")
                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    corr = df_user[numeric_cols].corr()
                    sns.heatmap(corr, annot=True, fmt=".2f",
                                cmap="coolwarm", ax=ax2,
                                center=0)
                    plt.tight_layout()
                    st.pyplot(fig2)

            if len(categorical_cols) > 0:
                st.markdown("**Categorical Column Distributions**")
                cols_to_plot = categorical_cols[:3]
                fig3, axes3 = plt.subplots(
                    1, len(cols_to_plot),
                    figsize=(14, 4))

                if len(cols_to_plot) == 1:
                    axes3 = [axes3]

                for ax, col in zip(axes3, cols_to_plot):
                    value_counts = df_user[col].value_counts().head(8)
                    ax.bar(range(len(value_counts)),
                           value_counts.values,
                           color="steelblue", alpha=0.7)
                    ax.set_xticks(range(len(value_counts)))
                    ax.set_xticklabels(
                        value_counts.index,
                        rotation=45, ha="right")
                    ax.set_title(col)

                plt.tight_layout()
                st.pyplot(fig3)

            # -- Anomaly detection if enough numeric columns --
            if len(numeric_cols) >= 2:
                st.subheader("Anomaly Detection")

                cols_for_anomaly = numeric_cols[:6]
                st.info(
                    f"Running anomaly detection on: "
                    f"{', '.join(cols_for_anomaly)}"
                )

                with st.spinner("Running anomaly detection..."):
                    X_user = df_user[cols_for_anomaly].dropna()
                    scaler_user = StandardScaler()
                    X_user_scaled = scaler_user.fit_transform(X_user)

                    iso_user = IsolationForest(
                        contamination=0.01,
                        random_state=42,
                        n_estimators=100
                    )
                    iso_user.fit(X_user_scaled)

                    predictions = iso_user.predict(X_user_scaled)
                    scores = iso_user.decision_function(X_user_scaled)

                    df_user = df_user.loc[X_user.index].copy()
                    df_user["Anomaly"] = predictions
                    df_user["Anomaly_Score"] = scores
                    df_user["Anomaly_Type"] = "Normal"
                    df_user.loc[
                        (df_user["Anomaly"] == -1),
                        "Anomaly_Type"] = "Risk"

                risk_count = len(df_user[df_user["Anomaly"] == -1])
                normal_count = len(df_user[df_user["Anomaly"] == 1])

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Normal Transactions", normal_count)
                with col2:
                    st.metric("Anomalies Detected", risk_count)

                # Scatter plot of first two numeric columns
                fig4, ax4 = plt.subplots(figsize=(10, 5))
                normal_mask = df_user["Anomaly"] == 1
                anomaly_mask = df_user["Anomaly"] == -1

                ax4.scatter(
                    df_user.loc[normal_mask, cols_for_anomaly[0]],
                    df_user.loc[normal_mask, cols_for_anomaly[1]],
                    color="steelblue", alpha=0.3,
                    s=10, label="Normal")
                ax4.scatter(
                    df_user.loc[anomaly_mask, cols_for_anomaly[0]],
                    df_user.loc[anomaly_mask, cols_for_anomaly[1]],
                    color="tomato", alpha=0.8,
                    s=30, label="Anomaly")

                ax4.set_xlabel(cols_for_anomaly[0])
                ax4.set_ylabel(cols_for_anomaly[1])
                ax4.set_title("Anomaly Detection Results")
                ax4.legend()
                plt.tight_layout()
                st.pyplot(fig4)

                # Show anomalous rows
                with st.expander("View Anomalous Rows"):
                    st.dataframe(
                        df_user[df_user["Anomaly"] == -1].head(20),
                        use_container_width=True
                    )

                # Update summary with anomaly findings
                summary_text += (
                    f" Anomaly detection found {risk_count} anomalous "
                    f"rows out of {len(df_user)} total rows."
                )

            # -- Build dynamic knowledge base and enable RAG --
            st.subheader("Ask Questions About Your Data")

            # Only rebuild if a new file was uploaded
            # Store filename in session_state to detect changes
            current_filename = uploaded_file.name

            if ("user_kb_filename" not in st.session_state or
                    st.session_state.user_kb_filename != current_filename):

                with st.spinner("Building knowledge base..."):
                    user_chunks, user_index = build_dynamic_knowledge_base(
                        df_user, summary_text)

                # Cache in session state
                st.session_state.user_chunks = user_chunks
                st.session_state.user_index = user_index
                st.session_state.user_kb_filename = current_filename
                st.session_state.user_data_messages = []

            else:
                user_chunks = st.session_state.user_chunks
                user_index = st.session_state.user_index

            st.success(
                f"Knowledge base ready with {len(user_chunks)} chunks. "
                f"Ask any question about your data below."
            )

            # Initialise session state for user data chat
            if "user_data_messages" not in st.session_state:
                st.session_state.user_data_messages = []

            # Display conversation history
            for message in st.session_state.user_data_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            user_data_question = st.chat_input(
                "Ask a question about your uploaded data...")

            if user_data_question:
                st.session_state.user_data_messages.append({
                    "role": "user",
                    "content": user_data_question
                })

                with st.chat_message("user"):
                    st.markdown(user_data_question)

                with st.chat_message("assistant"):
                    with st.spinner("Searching your data..."):
                        q_emb = embedding_model.encode(
                            [user_data_question])
                        q_emb = q_emb.astype(np.float32)
                        faiss.normalize_L2(q_emb)

                        D, I = user_index.search(q_emb, 3)

                        context = "\n\n".join([
                            user_chunks[i]["text"] for i in I[0]
                        ])

                        prompt = (
                            f"You are SentinelIQ, a business intelligence "
                            f"assistant. Use only the following data summary "
                            f"to answer the question. Be specific and use "
                            f"actual numbers.\n\n"
                            f"Context:\n{context}\n\n"
                            f"Question: {user_data_question}"
                        )

                        user_response = client.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=1000,
                            messages=[{
                                "role": "user",
                                "content": prompt
                            }]
                        )

                        user_answer = user_response.content[0].text

                    st.markdown(user_answer)

                st.session_state.user_data_messages.append({
                    "role": "assistant",
                    "content": user_answer
                })

        else:
            st.error(
                "Unsupported file format. Please upload a CSV, "
                "Excel, PDF, or image file."
            )