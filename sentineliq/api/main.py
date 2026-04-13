import json
import pickle
import numpy as np
import anthropic

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
# -------------------------------------------------------
# Initialise the FastAPI app
# This single object handles all incoming requests
# -------------------------------------------------------
app = FastAPI(title="SentinelIQ API", version="1.0")

# -------------------------------------------------------
# Load all saved models once at startup
# Loading here means every request reuses the same object
# rather than reloading from disk on every call
# -------------------------------------------------------
with open("models/fraud_model.pkl", "rb") as f:
    fraud_model = pickle.load(f)

with open("models/scaler_fraud.pkl", "rb") as f:
    scaler_fraud = pickle.load(f)

with open("models/anomaly_model.pkl", "rb") as f:
    anomaly_model = pickle.load(f)

with open("models/scaler_anomaly.pkl", "rb") as f:
    scaler_anomaly = pickle.load(f)

# Load RAG knowledge base
with open("knowledge_base/chunks.json", "r") as f:
    knowledge_base = json.load(f)

# Load FAISS index -- replaces raw embeddings and cosine similarity search
faiss_index = faiss.read_index("knowledge_base/faiss_index.bin")

# Load embedding model for query encoding
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialise Anthropic client for RAG answers
client = anthropic.Anthropic(api_key="Your_Anthropic_API_Key")

print("All models loaded successfully.")


# -------------------------------------------------------
# Pydantic models define the exact structure of each
# request and response. FastAPI validates every incoming
# request against these automatically before your code
# ever runs. If a field is missing or wrong type,
# FastAPI returns a clear error to the caller.
# -------------------------------------------------------

class FraudRequest(BaseModel):
    # All 30 features the fraud model expects
    # Time and Amount are the raw unscaled values
    # V1 to V28 are already PCA transformed
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

class FraudResponse(BaseModel):
    # What we send back after fraud prediction
    fraud_probability: float
    is_fraud: bool
    risk_level: str

class AnomalyRequest(BaseModel):
    # Four features the anomaly model expects
    Sales: float
    Quantity: float
    Discount: float
    Profit: float

class AnomalyResponse(BaseModel):
    # What we send back after anomaly detection
    anomaly_score: float
    anomaly_type: str
    is_anomaly: bool

class QueryRequest(BaseModel):
    # Plain English question for RAG pipeline
    question: str

class QueryResponse(BaseModel):
    # Answer from Claude plus sources used
    answer: str
    sources: list
    
    # -------------------------------------------------------
# Health check endpoint
# Always build this first -- it confirms the API is
# running before you test anything else
# -------------------------------------------------------
@app.get("/health")
def health_check():
    return {
        "status": "running",
        "models_loaded": True,
        "version": "1.0"
    }


# -------------------------------------------------------
# Fraud detection endpoint
# Accepts a transaction, returns fraud probability
# -------------------------------------------------------
@app.post("/predict/fraud", response_model=FraudResponse)
def predict_fraud(request: FraudRequest):
    # Convert request to numpy array in correct column order
    # Order must match exactly how the model was trained
    features = np.array([[
        request.Time, request.V1, request.V2, request.V3,
        request.V4, request.V5, request.V6, request.V7,
        request.V8, request.V9, request.V10, request.V11,
        request.V12, request.V13, request.V14, request.V15,
        request.V16, request.V17, request.V18, request.V19,
        request.V20, request.V21, request.V22, request.V23,
        request.V24, request.V25, request.V26, request.V27,
        request.V28, request.Amount
    ]])

    # Scale only Time and Amount -- columns 0 and 29
    features[:, [0, 29]] = scaler_fraud.transform(
        features[:, [0, 29]])

    # Get fraud probability -- column 1 is fraud class
    fraud_prob = fraud_model.predict_proba(features)[0][1]

    # Determine risk level based on probability thresholds
    if fraud_prob >= 0.8:
        risk_level = "HIGH"
    elif fraud_prob >= 0.5:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return FraudResponse(
        fraud_probability=round(float(fraud_prob), 4),
        is_fraud=fraud_prob >= 0.5,
        risk_level=risk_level
    )


# -------------------------------------------------------
# Anomaly detection endpoint
# Accepts a business transaction, returns anomaly type
# -------------------------------------------------------
@app.post("/predict/anomaly", response_model=AnomalyResponse)
def predict_anomaly(request: AnomalyRequest):
    # Build feature array in training order
    features = np.array([[
        request.Sales,
        request.Quantity,
        request.Discount,
        request.Profit
    ]])

    # Scale features using saved anomaly scaler
    features_scaled = scaler_anomaly.transform(features)

    # Get anomaly prediction and score
    prediction = anomaly_model.predict(features_scaled)[0]
    score = anomaly_model.decision_function(features_scaled)[0]

    # Determine anomaly type
    if prediction == 1:
        anomaly_type = "Normal"
    elif request.Profit > 0:
        anomaly_type = "Opportunity"
    else:
        anomaly_type = "Risk"

    return AnomalyResponse(
        anomaly_score=round(float(score), 4),
        anomaly_type=anomaly_type,
        is_anomaly=prediction == -1
    )


# -------------------------------------------------------
# RAG query endpoint
# Accepts a plain English question, returns AI answer
# -------------------------------------------------------
@app.post("/query", response_model=QueryResponse)
def query_sentineliq(request: QueryRequest):
    # Step 1 -- embed and normalise the question
    # Must normalise to match how index vectors were stored
    question_embedding = embedding_model.encode([request.question])
    question_embedding = question_embedding.astype(np.float32)
    faiss.normalize_L2(question_embedding)

    # Step 2 -- search FAISS index for top 3 relevant chunks
    # D = similarity scores, I = chunk indices
    D, I = faiss_index.search(question_embedding, 3)

    # Step 3 -- build context from retrieved chunks
    context = "\n\n".join([
        knowledge_base[i]["text"] for i in I[0]
    ])
    sources = [knowledge_base[i]["topic"] for i in I[0]]

    # Step 4 -- call Claude with context and question
    prompt = f"""You are SentinelIQ, a business intelligence assistant.
Use only the following business data to answer the question.
Be specific and use actual numbers from the context.

Context:
{context}

Question: {request.question}"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    return QueryResponse(
        answer=message.content[0].text,
        sources=sources
    )
    
    # -------------------------------------------------------
# Entry point -- runs the API server when you execute
# main.py directly from the terminal
# uvicorn is the server that handles incoming requests
# host 0.0.0.0 makes it accessible on your local network
# port 8000 is the standard port for local development
# reload=True restarts the server when you save changes
# -------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)