import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

ML_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ML_DIR)
for path in [PROJECT_ROOT, ML_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from ml.feature_engineering import FEATURE_COLUMNS, HUMAN_FACTOR_NAMES

ARTIFACTS_DIR = os.path.join(ML_DIR, "artifacts")

class FinShieldExplainer:
    def __init__(self, artifacts_dir=ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir
        self.xgb_model = XGBClassifier()
        self.xgb_model.load_model(os.path.join(artifacts_dir, "xgb_model.json"))
        
        self.iso_forest = joblib.load(os.path.join(artifacts_dir, "iso_forest.joblib"))
        self.scaler = joblib.load(os.path.join(artifacts_dir, "scaler.joblib"))
        
        with open(os.path.join(artifacts_dir, "feature_names.json"), "r") as f:
            self.feature_names = json.load(f)
            
        self.explainer = shap.TreeExplainer(self.xgb_model)

    def score_and_explain(self, df_single_unscaled):
        """
        Takes single-row pandas DataFrame containing raw unscaled features.
        Returns:
            risk_score: float [0.0, 1.0]
            decision: "ALLOW" | "STEP_UP_AUTH" | "BLOCK"
            top_factors: list of dicts [{"factor": str, "contribution": float}]
        """
        # Ensure exact column order
        X_raw = df_single_unscaled[self.feature_names].copy()
        X_scaled = self.scaler.transform(X_raw)
        
        # 1. XGBoost probability
        xgb_prob = float(self.xgb_model.predict_proba(X_scaled)[0, 1])
        
        # 2. Isolation Forest score
        raw_iso = float(self.iso_forest.score_samples(X_scaled)[0])
        iso_score = max(0.0, min(1.0, 1.0 - ((raw_iso - (-0.70)) / (0.35 + 1e-6))))
        
        # 3. Domain Rule Score (Heuristic telemetry check)
        domain_score = 0.0
        row = X_raw.iloc[0]
        if row.get("is_new_device", 0) == 1:
            domain_score += 0.30
        if row.get("location_deviation", 0) == 1:
            domain_score += 0.25
        if row.get("amount_z_score", 0) >= 3.0:
            domain_score += 0.35
        if row.get("hour_deviation", 0) >= 2:
            domain_score += 0.20
        if row.get("velocity_5m", 0) >= 2:
            domain_score += 0.25

        # 4. Blended Risk Score (Ensemble of XGBoost, IsoForest, and Domain telemetries)
        blended_val = 0.50 * xgb_prob + 0.25 * iso_score + 0.25 * min(1.0, domain_score)
        risk_score = round(min(1.0, max(0.0, blended_val)), 4)
        
        # 5. Decision Rule
        if risk_score < 0.30:
            decision = "ALLOW"
        elif risk_score <= 0.70:
            decision = "STEP_UP_AUTH"
        else:
            decision = "BLOCK"

        # 6. SHAP Factor Attribution
        shap_values = self.explainer.shap_values(X_scaled)[0]
        
        # Map features to SHAP values & human names
        factor_items = []
        for feat_name, s_val in zip(self.feature_names, shap_values):
            human_label = HUMAN_FACTOR_NAMES.get(feat_name, feat_name)
            feat_val = row.get(feat_name, 0)
            
            # Boost domain factors if explicitly flagged in telemetry
            weight = float(s_val)
            if feat_name == "amount_z_score" and feat_val >= 2.0:
                weight = max(weight, 0.45)
            elif feat_name == "is_new_device" and feat_val == 1:
                weight = max(weight, 0.40)
            elif feat_name == "location_deviation" and feat_val == 1:
                weight = max(weight, 0.30)
            elif feat_name == "hour_deviation" and feat_val >= 2:
                weight = max(weight, 0.25)
            elif feat_name == "velocity_5m" and feat_val >= 2:
                weight = max(weight, 0.35)

            if weight > 0.05:
                factor_items.append({
                    "raw_feature": feat_name,
                    "factor": human_label,
                    "shap_val": weight
                })

        # Sort by highest contribution
        factor_items.sort(key=lambda x: x["shap_val"], reverse=True)
        
        # Deduplicate factors
        unique_factors = []
        seen_names = set()
        for item in factor_items:
            if item["factor"] not in seen_names:
                unique_factors.append(item)
                seen_names.add(item["factor"])
                if len(unique_factors) >= 4:
                    break

        if not unique_factors:
            unique_factors = [
                {"factor": "Standard user location", "shap_val": 0.05},
                {"factor": "Known device fingerprint", "shap_val": 0.03}
            ]

        total_top_shap = sum(x["shap_val"] for x in unique_factors) or 1.0
        
        top_factors = []
        for x in unique_factors:
            pct_contrib = round(float(x["shap_val"] / total_top_shap), 2)
            top_factors.append({
                "factor": x["factor"],
                "contribution": pct_contrib
            })

        return {
            "risk_score": risk_score,
            "decision": decision,
            "xgb_prob": round(xgb_prob, 4),
            "iso_score": round(iso_score, 4),
            "top_factors": top_factors
        }
