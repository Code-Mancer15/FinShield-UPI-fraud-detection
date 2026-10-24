import numpy as np
import pandas as pd

# List of all feature names supplied to model
DOMAIN_FEATURES = [
    "amount_z_score",
    "is_new_device",
    "is_new_merchant",
    "location_deviation",
    "hour_deviation",
    "velocity_1m",
    "velocity_5m",
    "velocity_60m"
]

PCA_FEATURES = [f"V{i}" for i in range(1, 29)]

FEATURE_COLUMNS = DOMAIN_FEATURES + PCA_FEATURES

HUMAN_FACTOR_NAMES = {
    "amount_z_score": "Amount deviation",
    "is_new_device": "Unrecognized device",
    "is_new_merchant": "Unfamiliar merchant",
    "location_deviation": "Location anomaly",
    "hour_deviation": "Unusual transaction hour",
    "velocity_1m": "1-minute velocity burst",
    "velocity_5m": "5-minute velocity burst",
    "velocity_60m": "1-hour velocity burst",
}

for i in range(1, 29):
    HUMAN_FACTOR_NAMES[f"V{i}"] = f"Behavioral Pattern Signal (V{i})"

def extract_features_from_dataframe(df):
    """
    Extracts the design matrix X and target array y from a DataFrame.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df["is_fraud"].values if "is_fraud" in df.columns else None
    return X, y

def compute_realtime_features(txn_payload, user_profile, recent_user_txns):
    """
    Computes runtime feature vector for a single incoming transaction payload.
    txn_payload: dict containing amount, location, device_id, merchant_id, timestamp
    user_profile: dict/User object containing typical_amount_mean, typical_amount_std, home_city, devices, etc.
    recent_user_txns: list of recent transaction timestamps for velocity calculation
    """
    amount = float(txn_payload.get("amount", 0.0))
    mean_amt = float(user_profile.get("typical_amount_mean", 500.0))
    std_amt = max(1.0, float(user_profile.get("typical_amount_std", 200.0)))
    
    amount_z_score = round((amount - mean_amt) / std_amt, 2)
    
    # Device check
    known_devices = user_profile.get("known_device_ids", [])
    device_id = txn_payload.get("device_id", "")
    is_new_device = 0 if device_id in known_devices else 1
    
    # Merchant check
    known_merchants = user_profile.get("known_merchant_ids", [])
    merchant_id = txn_payload.get("merchant_id", "")
    is_new_merchant = 0 if merchant_id in known_merchants else 1
    
    # Location check
    home_city = user_profile.get("home_city", "")
    txn_location = txn_payload.get("location", "")
    location_deviation = 0 if txn_location.lower() == home_city.lower() else 1
    
    # Timestamp / hour check
    ts_str = txn_payload.get("timestamp")
    if ts_str:
        if isinstance(ts_str, str):
            try:
                dt = pd.to_datetime(ts_str)
            except Exception:
                dt = pd.Timestamp.now()
        else:
            dt = ts_str
    else:
        dt = pd.Timestamp.now()
        
    hour = dt.hour
    start_hour = int(user_profile.get("typical_hours_start", 8))
    end_hour = int(user_profile.get("typical_hours_end", 22))
    
    if start_hour <= hour <= end_hour:
        hour_deviation = 0
    else:
        hour_deviation = min(abs(hour - start_hour), abs(hour - end_hour))
        
    # Velocity calculation
    v_1m = sum(1 for t in recent_user_txns if (dt - t).total_seconds() <= 60)
    v_5m = sum(1 for t in recent_user_txns if (dt - t).total_seconds() <= 300)
    v_60m = sum(1 for t in recent_user_txns if (dt - t).total_seconds() <= 3600)
    
    # Construct feature row
    feature_dict = {
        "amount_z_score": amount_z_score,
        "is_new_device": is_new_device,
        "is_new_merchant": is_new_merchant,
        "location_deviation": location_deviation,
        "hour_deviation": hour_deviation,
        "velocity_1m": v_1m,
        "velocity_5m": v_5m,
        "velocity_60m": v_60m
    }
    
    # Add default/passed PCA features (if passed in payload, else 0.0)
    for i in range(1, 29):
        col = f"V{i}"
        feature_dict[col] = float(txn_payload.get(col, 0.0))
        
    df_single = pd.DataFrame([feature_dict])[FEATURE_COLUMNS]
    return df_single
