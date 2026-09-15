# 🌾 Agri-Market Intelligence Price Forecasting System

An end-to-end machine learning decision-support system and web application designed to forecast weekly agricultural commodity prices for Coimbatore markets using time-series feature engineering, FastAPI, and Streamlit.

---

## 🏗️ Project Architecture

text
Agri-intelligence/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application server
│   └── model.joblib         # Serialized Random Forest model
├── scripts/
│   └── export_model.py      # Feature engineering & training pipeline
├── tests/
│   └── test_api.py          # Integration test suite
├── dashboard.py             # Streamlit web UI client
├── start.ps1                # Orchestration script (Dual-process start)
├── requirements.txt         # Dependencies manifest
└── README.md