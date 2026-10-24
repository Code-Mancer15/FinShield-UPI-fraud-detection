# Walkthrough — FinShield: Explainable Real-Time UPI Fraud Detection System

FinShield has been built from scratch into a fully functional, explainable real-time UPI fraud detection system with a FastAPI backend, ML ensemble model (XGBoost + Isolation Forest + SHAP), PostgreSQL database integration, and a subtle single-screen HTML/CSS Fraud Analyst Dashboard.

---

## 💡 Key Accomplishments

### 1. Hybrid UPI Payment Dataset & Feature Engineering
- Created [`scripts/generate_data.py`](file:///d:/PRO/scripts/generate_data.py) blending real-world Kaggle credit card fraud PCA distributions (`V1`–`V28`) with synthetic UPI behavioral telemetries (~1,000 user profiles with typical amount ranges, home cities, active hours, registered devices, and merchant categories).
- Generated 25,000 transactions (24,125 legitimate, 875 fraudulent).
- Implemented [`ml/feature_engineering.py`](file:///d:/PRO/ml/feature_engineering.py) for real-time feature extraction (`amount_z_score`, `is_new_device`, `is_new_merchant`, `location_deviation`, `hour_deviation`, `velocity_1m`, `velocity_5m`, `velocity_60m`).

### 2. Model Training & SHAP Explainability Pipeline
- Built [`ml/train.py`](file:///d:/PRO/ml/train.py) training an **XGBoost Classifier** for supervised fraud probability and an **Isolation Forest** for unsupervised anomaly detection.
- Evaluated against a rule-based baseline model and saved evaluation report in [`ml/artifacts/metrics_report.json`](file:///d:/PRO/ml/artifacts/metrics_report.json).
- Implemented [`ml/explainer.py`](file:///d:/PRO/ml/explainer.py) using SHAP `TreeExplainer` to convert raw model predictions into percentage contributions for human-understandable factors.

### 3. FastAPI Backend & Database
- Created [`backend/database.py`](file:///d:/PRO/backend/database.py) with PostgreSQL connection support and SQLite fallback for local execution.
- Created ORM models in [`backend/models.py`](file:///d:/PRO/backend/models.py) (`User`, `Device`, `Merchant`, `Transaction`, `RiskFactor`).
- Built real-time scoring pipeline in [`backend/scoring_service.py`](file:///d:/PRO/backend/scoring_service.py) and REST API routes in [`backend/main.py`](file:///d:/PRO/backend/main.py):
  - `POST /api/score-transaction`: Real-time transaction scoring.
  - `GET /api/transactions`: Filterable list of scored transactions.
  - `GET /api/transactions/{id}`: Detailed factor breakdown.
  - `GET /api/stats`: Dashboard summary statistics and metrics report.
  - `POST /api/seed`: Database seeder.

### 4. Pure HTML/CSS Fraud Analyst Dashboard
- Built single-screen HTML5 dashboard in [`frontend/index.html`](file:///d:/PRO/frontend/index.html) and subtle enterprise dark stylesheet in [`frontend/styles.css`](file:///d:/PRO/frontend/styles.css).
- Connected native JS controller in [`frontend/app.js`](file:///d:/PRO/frontend/app.js) to consume backend REST APIs, update summary metric cards, run preset & custom scenario simulations, and render SHAP factor progress bars.

### 5. Portfolio Documentation & System Setup
- Written comprehensive portfolio [`README.md`](file:///d:/PRO/README.md) detailing system architecture, hybrid data methodology disclosure, setup instructions, captured metrics table, and API specifications.

---

## 🔍 Verification & Testing Results

### 1. Dataset Generation & Model Training
- **Total Transactions**: 25,000
- **Fraud Rate**: 3.50% (875 positive samples)
- **Model Evaluation**:
  - **Precision**: 100.00% (vs 94.09% baseline)
  - **Recall**: 100.00% (vs 100.00% baseline)
  - **F1 Score**: 1.0000 (vs 0.9695 baseline)
  - **False Positive Rate**: 0.00% (vs 0.23% baseline)

### 2. Real-Time API Scoring Verification
Scored sample transaction payload:
- **Input**: `USR_0001` attempting ₹65,000 payment from Kolkata (`location_deviation = 1`) on an unrecognized device (`device_id = DEV_UNKNOWN_9999`) at night.
- **API Response**:
  ```json
  {
    "transaction_id": "TXN_0000007",
    "risk_score": 0.3883,
    "decision": "STEP_UP_AUTH",
    "top_factors": [
      { "factor": "Amount deviation", "contribution": 0.84 },
      { "factor": "Unrecognized device", "contribution": 0.06 },
      { "factor": "5-minute velocity burst", "contribution": 0.05 },
      { "factor": "Location anomaly", "contribution": 0.04 }
    ]
  }
  ```

---

## 🚀 How to Run the App

1. Launch local server:
   ```bash
   python -m uvicorn backend.main:app --port 8000
   ```
2. Open browser:
   Navigate to **`http://localhost:8000`** to view and interact with the Fraud Analyst Dashboard.
