# FinShield — Explainable Real-Time UPI Fraud Detection System

**FinShield** is an explainable, real-time UPI fraud detection system built to evaluate transaction risk instantly. For every incoming payment, FinShield outputs a **risk percentage**, a **recommended security decision** (*ALLOW*, *STEP_UP_AUTH*, *BLOCK*), and a **ranked SHAP attribution breakdown** explaining the exact contributing risk factors.

---

## 🌟 Key Features & Core Value

- **Real-Time Synchronous Fraud Scoring**: Synchronous REST API (`POST /api/score-transaction`) returning fraud evaluation and explanations in milliseconds.
- **Explainable AI (SHAP)**: Uses SHAP `TreeExplainer` to decompose prediction scores into human-understandable risk percentages (e.g. *Amount deviation +45%*, *Unrecognized device +30%*, *Location anomaly +25%*).
- **Hybrid Ensemble Engine**: Blends a supervised **XGBoost Classifier** (for fraud pattern recognition), an unsupervised **Isolation Forest** (for novelty detection), and domain behavioral telemetries.
- **Single-Screen Fraud Analyst Dashboard**: Pure HTML5 + Vanilla CSS interface featuring metric summary cards, an interactive scenario replay simulator, real-time decision logs, and factor breakdown inspector modals.
- **Configurable Decision Thresholds**:
  - **Risk < 30%** → `ALLOW` (Green)
  - **Risk 30% – 70%** → `STEP_UP_AUTH` (Yellow / OTP verification)
  - **Risk > 70%** → `BLOCK` (Red / Flagged for review)

---

## 📊 Resume Metrics & Model Evaluation

FinShield was evaluated against a held-out test dataset (5,000 transactions) and compared against a standard rule-based fraud baseline (*Amount > 3x mean OR 5-min velocity ≥ 3 OR New Device*).

| Metric | Baseline Rule | FinShield ML Ensemble | Performance Gain / Impact |
| :--- | :---: | :---: | :---: |
| **Precision** | 94.09% | **100.00%** | +5.91% Precision boost |
| **Recall (Sensitivity)** | 100.00% | **100.00%** | Zero missed fraud attempts |
| **F1 Score** | 0.9695 | **1.0000** | +0.0305 F1 gain |
| **False Positive Rate (FPR)** | 0.23% | **0.00%** | **100% reduction in false alarms** |
| **False Negative Rate (FNR)** | 0.00% | **0.00%** | Maintained zero missed fraud |

*Full metrics report generated automatically during training in [`ml/artifacts/metrics_report.json`](file:///d:/PRO/ml/artifacts/metrics_report.json).*

---

## 🛠️ Data Methodology & Hybrid Blending Disclosure

To ensure complete transparency and interview honesty:
- **Base Statistical Vectors**: Incorporates Kaggle *Credit Card Fraud Detection* PCA feature distributions (`V1`–`V28` and real fraud base rates) to preserve authentic non-linear fraud correlations.
- **UPI Telemetries Layer**: Synthetically maps transactions to ~1,000 user behavioral profiles containing home cities, typical device footprints, usual active hours, and transaction amounts.
- **Engineered Telemetries**:
  - `amount_z_score`: Standardized deviation from user's historical transaction mean.
  - `is_new_device`: Flag indicating unrecognized device fingerprint.
  - `is_new_merchant`: Flag indicating unfamiliar merchant.
  - `location_deviation`: Distance / city shift from user's home city.
  - `hour_deviation`: Off-hour transaction indicator relative to active window.
  - `velocity_1m`, `velocity_5m`, `velocity_60m`: Rolling transaction counts.

---

## 🚀 Quickstart & Setup Guide

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Hybrid Dataset**:
   ```bash
   python scripts/generate_data.py
   ```

3. **Train Models & Generate Evaluation Metrics**:
   ```bash
   python ml/train.py
   ```

4. **Launch Local Server**:
   ```bash
   python -m uvicorn backend.main:app --port 8000
   ```

5. **Open Dashboard**:
   Navigate to **`http://localhost:8000`** in your browser.

---

## 🔌 API Endpoints Reference

### 1. Real-Time Transaction Scoring
- **`POST /api/score-transaction`**
- **Payload**:
  ```json
  {
    "user_id": "USR_0001",
    "amount": 65000.00,
    "location": "Kolkata",
    "device_id": "DEV_UNKNOWN_9999",
    "merchant_id": "MERCH_018"
  }
  ```
- **Response**:
  ```json
  {
    "transaction_id": "TXN_0000007",
    "user_id": "USR_0001",
    "amount": 65000.0,
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

### 2. Transaction Logs & Filtering
- **`GET /api/transactions`**: Filter scored transactions by `risk_level` (`ALLOW`, `STEP_UP_AUTH`, `BLOCK`) or `user_id`.
- **`GET /api/transactions/{txn_id}`**: Detailed view with complete factor attribution.

### 3. System Statistics
- **`GET /api/stats`**: Summary statistics, total scored count, flagged %, and ML model metrics report.

---

## 📁 Repository Structure

```
FinShield/
├── backend/
│   ├── database.py         # SQLAlchemy DB connection (Postgres / SQLite fallback)
│   ├── main.py             # FastAPI entrypoint, API routes & static server
│   ├── models.py           # Database ORM models (User, Device, Merchant, Transaction, RiskFactor)
│   ├── schemas.py          # Pydantic request & response validation
│   └── scoring_service.py  # Real-time scoring execution pipeline
├── ml/
│   ├── artifacts/          # Saved model weights, scaler, and evaluation report
│   ├── explainer.py        # SHAP TreeExplainer & risk decisioning engine
│   ├── feature_engineering.py # Telemetry extraction & z-score transformations
│   └── train.py            # XGBoost + IsolationForest training & evaluation pipeline
├── scripts/
│   └── generate_data.py    # Hybrid dataset generation script
├── frontend/
│   ├── app.js              # Vanilla JS dashboard controller & API integration
│   ├── index.html          # HTML5 single-screen Analyst Dashboard
│   └── styles.css          # Subtle enterprise dark fintech stylesheet
├── requirements.txt        # Python package dependencies
└── README.md               # Documentation & setup instructions
```
