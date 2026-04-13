# SentinelIQ

An end-to-end autonomous business intelligence and risk detection platform. SentinelIQ combines time-series analysis, unsupervised anomaly detection, supervised fraud detection, and a retrieval-augmented generation pipeline into a single deployable system. Users can upload any business dataset and receive automated insights, anomaly flags, and plain English answers to questions about their data.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the Application](#running-the-application)
- [Running the API](#running-the-api)
- [Running with Docker](#running-with-docker)
- [Notebooks](#notebooks)
- [Model Details](#model-details)
- [Dataset Sources](#dataset-sources)
- [Results](#results)
- [Portfolio Notes](#portfolio-notes)

---

## Project Overview

SentinelIQ was built to answer the four questions a business intelligence system should always be able to answer:

- What happened?
- Why did it happen?
- Is it dangerous?
- What will happen next?

The system processes structured data (CSV, Excel), unstructured data (PDF reports), and visual data (chart images) through a unified pipeline that produces automated insights, anomaly detection results, and natural language answers powered by Claude AI.

---

## Features

### Business Intelligence Dashboard
- Monthly sales and profit trend analysis from 2014 to 2017
- Rolling average smoothing to separate signal from noise
- Profit margin analysis by product category and geographic region
- Automated rule-based insight generation with severity classification

### Anomaly Detection
- Isolation Forest unsupervised anomaly detection
- Anomaly scoring with configurable contamination threshold
- Separation of risk anomalies (unusual losses) from opportunity anomalies (unusual gains)
- Visual scatter plot showing flagged transactions in context

### Fraud Detection
- Logistic Regression classifier trained on 284,807 credit card transactions
- SMOTE oversampling to handle severe class imbalance (0.17% fraud rate)
- Fraud probability scoring with three-tier risk classification
- Visual probability gauge for each transaction

### RAG Query System
- Sentence embeddings generated using all-MiniLM-L6-v2
- FAISS vector index for scalable similarity search
- Dynamic knowledge base generated automatically from dataset findings
- Plain English querying powered by Claude API
- Conversation history with source attribution

### Analyse Your Data
- Upload any CSV, Excel, PDF, or chart image
- Automatic EDA including distributions, correlation heatmap, and summary statistics
- Anomaly detection runs automatically on any numeric dataset
- Dynamic knowledge base built from uploaded data for RAG querying
- PDF text extraction and AI-powered report summarisation
- Claude vision API integration for chart and dashboard interpretation

---

## Tech Stack

### Machine Learning
- scikit-learn -- Isolation Forest, Logistic Regression, StandardScaler, train-test split
- imbalanced-learn -- SMOTE oversampling for class imbalance
- sentence-transformers -- all-MiniLM-L6-v2 for text embeddings
- FAISS -- vector index for similarity search

### Backend
- FastAPI -- REST API framework
- uvicorn -- ASGI server
- pydantic -- request and response validation
- pickle -- model serialisation

### Frontend
- Streamlit -- five-page interactive web application
- matplotlib -- trend charts, scatter plots, histograms
- seaborn -- correlation heatmap

### AI
- Anthropic Claude API -- RAG answer generation and vision-based chart interpretation

### Data Processing
- pandas -- data loading, cleaning, aggregation
- numpy -- numerical operations
- pdfplumber -- PDF text extraction
- Pillow -- image handling

### Deployment
- Docker -- containerisation
- uvicorn -- production server inside container

---

## Project Structure

```
sentineliq/
    data/
        Sample - Superstore.csv
        creditcard.csv

    notebooks/
        01_eda.ipynb
        02_anomaly_detection.ipynb
        03_sentineliq_alerts.ipynb
        04_rag_query.ipynb
        05_fraud_detection.ipynb
        06_save_models.ipynb
        07_faiss_upgrade.ipynb

    api/
        main.py
        Dockerfile
        requirements.txt
        .dockerignore
        models/
            fraud_model.pkl
            scaler_fraud.pkl
            anomaly_model.pkl
            scaler_anomaly.pkl
        knowledge_base/
            chunks.json
            embeddings.npy
            faiss_index.bin

    app/
        sentineliq_app.py
```

---

## Setup and Installation

### Prerequisites
- Python 3.10 or higher
- pip
- Docker Desktop (for containerised deployment)
- Anthropic API key

### Clone the repository

```bash
git clone https://github.com/yourusername/sentineliq.git
cd sentineliq
```

### Install dependencies

```bash
pip install fastapi uvicorn pydantic pandas numpy scikit-learn imbalanced-learn sentence-transformers faiss-cpu anthropic streamlit matplotlib seaborn pdfplumber openpyxl Pillow
```

### Download datasets

The following datasets are required and must be placed in the `data/` folder:

- Superstore Sales Dataset: https://www.kaggle.com/datasets/vivek468/superstore-dataset-final
- Credit Card Fraud Detection: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Set your API key

In both `api/main.py` and `app/sentineliq_app.py`, replace the placeholder with your Anthropic API key:

```python
client = anthropic.Anthropic(api_key="your_api_key_here")
```

### Train and save models

Run all cells in `notebooks/06_save_models.ipynb` to train the models and save them to the `api/models/` and `api/knowledge_base/` directories. Then run `notebooks/07_faiss_upgrade.ipynb` to build the FAISS index.

---

## Running the Application

### Streamlit frontend

```bash
cd app
streamlit run sentineliq_app.py
```

The app will open automatically at `http://localhost:8501`.

---

## Running the API

```bash
cd api
python main.py
```

The API runs at `http://localhost:8000`. Interactive documentation is available at `http://localhost:8000/docs`.

### Available endpoints

```
GET  /health           -- health check
POST /predict/fraud    -- fraud probability for a credit card transaction
POST /predict/anomaly  -- anomaly score for a business transaction
POST /query            -- plain English question answered via RAG pipeline
```

---

## Running with Docker

### Build the container

```bash
cd api
docker build -t sentineliq-api .
```

### Run the container

```bash
docker run -p 8000:8000 sentineliq-api
```

The API will be available at `http://localhost:8000`.

---

## Notebooks

| Notebook | Description |
|---|---|
| 01_eda.ipynb | Time-series EDA, rolling averages, business metrics, automated insight generation |
| 02_anomaly_detection.ipynb | Isolation Forest training, scoring, risk vs opportunity separation |
| 03_sentineliq_alerts.ipynb | Combined alert system with regional and category severity ranking |
| 04_rag_query.ipynb | Embedding generation, cosine similarity search, Claude API integration |
| 05_fraud_detection.ipynb | Class imbalance analysis, SMOTE, Logistic Regression, ROC-AUC evaluation |
| 06_save_models.ipynb | Model serialisation, knowledge base generation, embedding computation |
| 07_faiss_upgrade.ipynb | FAISS index construction and validation |

---

## Model Details

### Anomaly Detection -- Isolation Forest
- Algorithm: Isolation Forest
- Features: Sales, Quantity, Discount, Profit
- Contamination: 0.01 (1% expected anomaly rate)
- Scaler: StandardScaler applied to all four features
- Output: Anomaly score and classification (Normal, Risk, Opportunity)

### Fraud Detection -- Logistic Regression
- Algorithm: Logistic Regression with SMOTE oversampling
- Features: Time, V1-V28 (PCA transformed), Amount
- Training samples after SMOTE: 454,902
- Scaler: StandardScaler applied to Time and Amount only
- Output: Fraud probability and three-tier risk classification

### RAG Pipeline
- Embedding model: all-MiniLM-L6-v2 (384 dimensions)
- Vector index: FAISS IndexFlatIP with L2 normalisation
- Retrieval: top-3 chunks per query
- Generation: Claude claude-sonnet-4-20250514

---

## Dataset Sources

- Superstore Sales: Kaggle -- vivek468/superstore-dataset-final
  - 9,994 retail transactions, 2014 to 2017, 21 features
- Credit Card Fraud: Kaggle -- mlg-ulb/creditcardfraud
  - 284,807 transactions, 492 fraud cases, 30 features

---

## Results

| Metric | Value |
|---|---|
| Fraud detection recall | 0.92 |
| Fraud detection ROC-AUC | 0.97 |
| Baseline recall (no SMOTE) | 0.64 |
| Fraud cases missed (baseline) | 35 of 98 |
| Fraud cases missed (SMOTE model) | 8 of 98 |
| Anomalies detected in Superstore | 100 of 9,994 transactions |
| Risk anomalies | 32 |
| Opportunity anomalies | 68 |
| Most anomalous transaction | January 2015, Furniture, South region, profit of -1862.31 on a 4297.64 sale |

---

## Portfolio Notes

This project demonstrates:

- Unsupervised machine learning -- Isolation Forest with no labelled examples
- Handling class imbalance -- SMOTE oversampling on a 0.17% minority class
- Evaluation beyond accuracy -- Recall, Precision, F1, ROC-AUC on imbalanced data
- RAG architecture -- embeddings, vector search, and grounded LLM generation
- Full-stack ML engineering -- notebooks to API to frontend to container
- Multimodal AI -- structured data analysis, PDF processing, and vision-based chart interpretation
- Production practices -- model serialisation, data leakage prevention, StandardScaler fit only on training data, stratified train-test split

---

## License

MIT License. See LICENSE for details.