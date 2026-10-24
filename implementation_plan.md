# Implementation Plan — FinShield: Explainable Real-Time UPI Fraud Detection System

FinShield is an explainable real-time UPI fraud detection system built with Python, FastAPI, PostgreSQL, XGBoost, Isolation Forest, SHAP, and a clean vanilla HTML/CSS/JS Fraud Analyst Dashboard.

## User Review Required

> [!IMPORTANT]
> **Data & ML Blending Architecture**:
> - **Data Generation**: A standalone script (`scripts/generate_data.py`) will generate synthetic UPI user profiles (~1000 users) and synthetic transaction logs with realistic behavior shifts (amount z-score, location deviation, new device, velocity bursts, odd-hour transactions) combined with Kaggle-style PCA features.
> - **Model Blend & Explainability**: Final Risk % is computed as `Risk = (0.75 * XGB_Prob + 0.25 * IsoForest_Anomaly_Score) * 100`. SHAP TreeExplainer will evaluate feature attribution on the XGBoost model and normalize factor percentages for human readability.
> - **Decision Thresholds**:
>   - Risk < 30%: **ALLOW** (Green)
>   - Risk 30%–70%: **STEP_UP_AUTH** (Yellow/Amber OTP requirement)
>   - Risk > 70%: **BLOCK** (Red)

## Open Questions

None at this time. Requirements are clear and concise.

---

## Proposed Changes

### Component 1: Data Generator & Feature Engineering Pipeline
Directory: `ml/` & `scripts/`

#### [NEW] [`scripts/generate_data.py`](file:///d:/PRO/scripts/generate_data.py)
- Generates 1,000 synthetic users with behavioral distributions (typical amount min/max/mean/std, home city, usual active hours range, registered devices, frequent merchants).
- Generates ~25,000 synthetic UPI transactions including normal transactions and injected fraud scenarios (velocity attacks, account takeover via new device + high amount, location anomaly + odd hour).
- Saves synthetic raw transactions to `data/upi_transactions.csv`.

#### [NEW] [`ml/feature_engineering.py`](file:///d:/PRO/ml/feature_engineering.py)
- Computes user-level rolling profiles (historical mean amount, historical std, known devices, home location, hourly transaction counts in last 1m, 5m, 60m).
- Engineers model features: `amount_z_score`, `is_new_device`, `is_new_merchant`, `location_distance_km`, `hour_deviation`, `velocity_1m`, `velocity_5m`, `velocity_60m`, and anonymized PCA features.

---

### Component 2: Model Training, SHAP & Evaluation Pipeline

#### [NEW] [`ml/train.py`](file:///d:/PRO/ml/train.py)
- Splits data into train (80%) and test (20%).
- Trains `XGBClassifier` for supervised fraud probability.
- Trains `IsolationForest` on legitimate transactions for unsupervised novelty detection.
- Evaluates against a rule-based baseline (e.g. `amount > 3x mean OR velocity_5m > 3`).
- Generates evaluation report (`ml/metrics_report.json`) containing Precision, Recall, F1 Score, False Positive Rate (FPR), and False Negative Rate (FNR) for both baseline and FinShield ensemble model.
- Saves trained models (`xgb_model.json`, `iso_forest.joblib`) and scaler to `ml/artifacts/`.

#### [NEW] [`ml/explainer.py`](file:///d:/PRO/ml/explainer.py)
- Initializes SHAP `TreeExplainer` on the trained XGBoost model.
- Computes per-transaction SHAP values and transforms raw log-odds contributions into human-readable percentage contributions (e.g. Amount deviation: +25%, New device: +20%).

---

### Component 3: Database Schema & Backend API (FastAPI)

#### [NEW] [`backend/database.py`](file:///d:/PRO/backend/database.py)
- SQLAlchemy database connection and session management with SQLite default for standalone local execution (or PostgreSQL via DATABASE_URL).

#### [NEW] [`backend/models.py`](file:///d:/PRO/backend/models.py)
- Declarative ORM models: `User`, `Device`, `Merchant`, `Transaction`, `RiskFactor`.

#### [NEW] [`backend/schemas.py`](file:///d:/PRO/backend/schemas.py)
- Pydantic request/response schemas for `/score-transaction`, `/transactions`, `/stats`, `/users/{id}/profile`.

#### [NEW] [`backend/scoring_service.py`](file:///d:/PRO/backend/scoring_service.py)
- End-to-end real-time transaction pipeline: fetching user profile, computing rolling window velocity, extracting features, running model prediction + SHAP explanation, saving record to Postgres DB, and returning structured JSON response.

#### [NEW] [`backend/main.py`](file:///d:/PRO/backend/main.py)
- FastAPI application entry point.
- Configures CORS, mounts static frontend directory, initializes DB tables on startup, registers API routes:
  - `POST /api/score-transaction`: Real-time transaction scoring.
  - `GET /api/transactions`: Filterable list of scored transactions.
  - `GET /api/transactions/{id}`: Detailed transaction breakdown.
  - `GET /api/stats`: Dashboard summary statistics and ML evaluation metrics.
  - `POST /api/seed`: Seed initial user profiles & sample transaction data.
  - `GET /api/sample-scenarios`: Pre-packaged fraud & legit test scenarios for interactive dashboard testing.

---

### Component 4: Fraud Analyst Dashboard (Frontend)

#### [NEW] [`frontend/index.html`](file:///d:/PRO/frontend/index.html)
- Clean, professional single-screen dashboard layout:
  - **Top Navigation & Metric Cards**: Total Scored, Flagged Rate, Model Precision/Recall vs Baseline banner.
  - **Transaction Replay & Simulator Panel**: Select sample test scenario or customize payload (Amount, Merchant, Device, Location) and click "Score Transaction" to watch real-time risk decisioning.
  - **Recent Scored Transactions Table**: Risk score badges (Green/Yellow/Red), user ID, amount, timestamp, decision action.
  - **Factor Breakdown Inspector Modal/Panel**: Visual SHAP contribution breakdown bar chart showing top factors and contribution percentages.

#### [NEW] [`frontend/styles.css`](file:///d:/PRO/frontend/styles.css)
- Custom Vanilla CSS: Subtle dark-mode fintech design, crisp typography (Inter font stack), clean CSS grid layout, distinct risk badges, sleek progress bars for factor contributions. No overly flashy animations.

#### [NEW] [`frontend/app.js`](file:///d:/PRO/frontend/app.js)
- Vanilla JS API handler: Fetch `/api/stats`, `/api/transactions`, `/api/score-transaction`, update DOM dynamically.

---

### Component 5: Packaging & Portfolio Documentation

#### [NEW] [`requirements.txt`](file:///d:/PRO/requirements.txt)
- Dependencies: `fastapi`, `uvicorn`, `sqlalchemy`, `pydantic`, `xgboost`, `scikit-learn`, `shap`, `pandas`, `numpy`, `joblib`.

#### [NEW] [`README.md`](file:///d:/PRO/README.md)
- Portfolio README with system architecture diagram, setup instructions (local Python execution), synthetic data methodology disclosure, API specification, captured model metrics table (Precision, Recall, F1, FNR reduction vs baseline), and screenshot placeholders.

---

## Verification Plan

### Automated Verification
1. **Data Generation & Training**:
   - Run `python scripts/generate_data.py` to create dataset.
   - Run `python ml/train.py` to train XGBoost + Isolation Forest models and generate `ml/metrics_report.json`.
   - Verify non-zero metrics and confirmed precision/recall output.
2. **API & Scoring Service Tests**:
   - Run pytest or inline test script testing `/api/score-transaction` with standard and anomalous payloads.
   - Verify returned JSON structure: `risk_score`, `decision` (ALLOW, STEP_UP_AUTH, BLOCK), and top SHAP factors breakdown.

### Manual Verification
1. Launch local backend server (`python -m uvicorn backend.main:app --port 8000`).
2. Open `http://localhost:8000` in browser.
3. Test replay/simulation panel: score normal transaction -> verify ALLOW decision (<30%). Score high-amount/new-device transaction -> verify BLOCK (>70%) with top SHAP factor breakdown.
4. Verify stats cards update accurately.
