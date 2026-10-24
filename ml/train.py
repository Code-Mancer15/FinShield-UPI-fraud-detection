import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

import sys
ML_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ML_DIR)
for path in [PROJECT_ROOT, ML_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from ml.feature_engineering import FEATURE_COLUMNS, extract_features_from_dataframe

ML_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(ML_DIR), "data", "upi_transactions.csv")
ARTIFACTS_DIR = os.path.join(ML_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def train_and_evaluate():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    X, y = extract_features_from_dataframe(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Train size: {len(X_train)} (Fraud: {y_train.sum()}), Test size: {len(X_test)} (Fraud: {y_test.sum()})")
    
    # 1. Scaler for features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. XGBoost Supervised Classifier
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    xgb_model = XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )
    xgb_model.fit(X_train_scaled, y_train)
    
    # XGBoost raw probabilities
    xgb_test_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]
    
    # 3. Isolation Forest (Unsupervised on legitimate training transactions only)
    X_train_legit = X_train_scaled[y_train == 0]
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.035,
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_train_legit)
    
    # Isolation forest raw decision function (lower means more anomalous)
    raw_iso_scores = iso_forest.score_samples(X_test_scaled)
    # Scale score to [0, 1] range where 1 is highest anomaly risk
    min_s, max_s = raw_iso_scores.min(), raw_iso_scores.max()
    iso_test_scores = 1.0 - ((raw_iso_scores - min_s) / (max_s - min_s + 1e-6))
    
    # 4. Ensemble Risk Score (75% XGBoost + 25% Isolation Forest)
    ensemble_risk_scores = (0.75 * xgb_test_probs + 0.25 * iso_test_scores)
    model_preds = (ensemble_risk_scores >= 0.30).astype(int)
    
    # 5. Baseline Rule Model
    # Rule: Flag if amount z-score > 3.0 OR velocity_5m >= 3 OR is_new_device == 1
    baseline_preds = (
        (X_test["amount_z_score"] > 3.0) | 
        (X_test["velocity_5m"] >= 3) | 
        (X_test["is_new_device"] == 1)
    ).astype(int).values

    # 6. Calculate Metrics
    def get_metrics_dict(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        return {
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "false_positive_rate": round(float(fpr), 4),
            "false_negative_rate": round(float(fnr), 4)
        }

    baseline_metrics = get_metrics_dict(y_test, baseline_preds)
    model_metrics = get_metrics_dict(y_test, model_preds)
    
    fnr_reduction = round(((baseline_metrics["false_negative_rate"] - model_metrics["false_negative_rate"]) / 
                           max(1e-6, baseline_metrics["false_negative_rate"])) * 100, 2)
    
    metrics_report = {
        "dataset": {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "fraud_count_test": int(y_test.sum())
        },
        "baseline_rule": baseline_metrics,
        "finshield_ensemble": model_metrics,
        "improvements": {
            "fnr_reduction_pct": fnr_reduction,
            "f1_gain": round(model_metrics["f1_score"] - baseline_metrics["f1_score"], 4)
        }
    }
    
    print("\n--- MODEL EVALUATION REPORT ---")
    print(f"Baseline Rule    -> Precision: {baseline_metrics['precision']:.4f}, Recall: {baseline_metrics['recall']:.4f}, F1: {baseline_metrics['f1_score']:.4f}, FNR: {baseline_metrics['false_negative_rate']:.4f}")
    print(f"FinShield Ensemble -> Precision: {model_metrics['precision']:.4f}, Recall: {model_metrics['recall']:.4f}, F1: {model_metrics['f1_score']:.4f}, FNR: {model_metrics['false_negative_rate']:.4f}")
    print(f"False Negative Rate Reduction: {fnr_reduction}%")
    
    # 7. Save Artifacts
    xgb_model.save_model(os.path.join(ARTIFACTS_DIR, "xgb_model.json"))
    joblib.dump(iso_forest, os.path.join(ARTIFACTS_DIR, "iso_forest.joblib"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.joblib"))
    
    with open(os.path.join(ARTIFACTS_DIR, "metrics_report.json"), "w") as f:
        json.dump(metrics_report, f, indent=2)
        
    with open(os.path.join(ARTIFACTS_DIR, "feature_names.json"), "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)
        
    print(f"Saved artifacts to {ARTIFACTS_DIR}")

if __name__ == "__main__":
    train_and_evaluate()
