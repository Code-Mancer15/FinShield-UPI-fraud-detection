import os
import json
import random
import math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

CITIES = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Surat"]

MERCHANT_SPECS = [
    ("Swiggy", "Food", 0.1),
    ("Zomato", "Food", 0.1),
    ("Amazon Pay", "Retail", 0.2),
    ("Flipkart", "Retail", 0.2),
    ("Uber UPI", "Travel", 0.15),
    ("Ola Cabs", "Travel", 0.15),
    ("BigBasket", "Grocery", 0.1),
    ("Reliance Smart", "Grocery", 0.1),
    ("D-Mart UPI", "Grocery", 0.1),
    ("HPCL Petrol Pump", "Fuel", 0.2),
    ("Indian Oil Petrol", "Fuel", 0.2),
    ("Local Kirana Store", "Retail", 0.05),
    ("Tea & Snacks Corner", "Food", 0.05),
    ("Croma Electronics", "Electronics", 0.6),
    ("Vijay Sales", "Electronics", 0.6),
    ("Tanishq Jewelry", "Jewelry", 0.85),
    ("Kalyan Jewellers", "Jewelry", 0.85),
    ("P2P Contact Transfer", "P2P", 0.3),
    ("Self Account Transfer", "P2P", 0.05),
    ("International Remittance", "P2P", 0.75)
]

FIRST_NAMES = ["Aarav", "Ananya", "Rohan", "Priya", "Vikram", "Neha", "Aditya", "Pooja", "Rahul", "Kavya", 
               "Siddharth", "Isha", "Arjun", "Sneha", "Karan", "Riya", "Amit", "Meera", "Manish", "Divya"]
LAST_NAMES = ["Sharma", "Verma", "Patel", "Gupta", "Singh", "Rao", "Nair", "Joshi", "Kulkarni", "Mehta", 
              "Deshmukh", "Chopra", "Reddy", "Banerjee", "Agarwal", "Iyer", "Jain", "Shah", "Bhat", "Saxena"]

def generate_user_profiles(num_users=1000):
    users = []
    devices = []
    
    for i in range(1, num_users + 1):
        user_id = f"USR_{i:04d}"
        name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        home_city = random.choice(CITIES)
        mean_amt = round(random.uniform(150, 2500), 2)
        std_amt = round(mean_amt * random.uniform(0.2, 0.5), 2)
        
        # Typical active hours e.g. 08:00 to 22:00
        start_hour = random.choice([7, 8, 9, 10])
        end_hour = random.choice([20, 21, 22, 23])
        
        # Primary device fingerprint
        primary_dev_id = f"DEV_{user_id}_01"
        devices.append({
            "id": primary_dev_id,
            "user_id": user_id,
            "device_fingerprint": f"fp_{random.randint(100000, 999999)}",
            "first_seen_at": (datetime.now() - timedelta(days=random.randint(60, 365))).isoformat()
        })
        
        users.append({
            "id": user_id,
            "name": name,
            "home_city": home_city,
            "typical_amount_mean": mean_amt,
            "typical_amount_std": std_amt,
            "typical_hours_start": start_hour,
            "typical_hours_end": end_hour,
            "primary_device_id": primary_dev_id
        })
        
    return users, devices

def generate_merchants():
    merchants = []
    for idx, (name, category, risk_w) in enumerate(MERCHANT_SPECS, 1):
        m_id = f"MERCH_{idx:03d}"
        merchants.append({
            "id": m_id,
            "name": name,
            "category": category,
            "risk_weight": risk_w
        })
    return merchants

def generate_kaggle_pca_vectors(num_samples, is_fraud):
    """
    Generates synthetic V1..V28 vectors matching Kaggle Credit Card dataset statistics.
    Fraud samples (is_fraud=1) have shifts in V4, V11, V12, V14, V17.
    """
    V = {}
    for i in range(1, 29):
        if is_fraud:
            # Significant distribution shift on known Kaggle fraud predictor components
            if i in [4, 11]:
                mean_val = np.random.uniform(2.5, 6.0)
            elif i in [12, 14, 17]:
                mean_val = np.random.uniform(-7.0, -3.0)
            elif i in [10, 16]:
                mean_val = np.random.uniform(-4.0, -1.5)
            else:
                mean_val = np.random.normal(0, 1.2)
            std_val = np.random.uniform(1.0, 2.5)
        else:
            mean_val = 0.0
            std_val = 1.0
            
        V[f"V{i}"] = round(float(np.random.normal(mean_val, std_val)), 4)
    return V

def generate_transactions(users, merchants, devices, total_txns=20000, fraud_ratio=0.03):
    user_dict = {u["id"]: u for u in users}
    merchant_list = merchants
    device_by_user = {}
    for d in devices:
        device_by_user.setdefault(d["user_id"], []).append(d["id"])
        
    transactions = []
    base_time = datetime.now() - timedelta(days=30)
    
    # Keep track of user transaction timestamps for rolling velocity computation
    user_txn_timestamps = {u["id"]: [] for u in users}
    
    num_fraud = int(total_txns * fraud_ratio)
    num_legit = total_txns - num_fraud
    
    # Create target array
    labels = [0] * num_legit + [1] * num_fraud
    random.shuffle(labels)
    
    current_time = base_time
    
    for idx, is_fraud in enumerate(labels, 1):
        # Progress time by 1-5 minutes per step
        current_time += timedelta(seconds=random.randint(30, 240))
        user = random.choice(users)
        u_id = user["id"]
        
        # Calculate recent velocity for this user before adding current txn
        past_timestamps = user_txn_timestamps[u_id]
        v_1m = sum(1 for t in past_timestamps if (current_time - t).total_seconds() <= 60)
        v_5m = sum(1 for t in past_timestamps if (current_time - t).total_seconds() <= 300)
        v_60m = sum(1 for t in past_timestamps if (current_time - t).total_seconds() <= 3600)
        
        if is_fraud == 0:
            # Normal UPI transaction behavior
            # Amount around typical user mean
            amount = max(10.0, round(np.random.normal(user["typical_amount_mean"], user["typical_amount_std"]), 2))
            
            # Location = user's home city (95%) or travel city (5%)
            location = user["home_city"] if random.random() < 0.95 else random.choice(CITIES)
            
            # Device = registered device
            device_id = device_by_user[u_id][0]
            is_new_device = 0
            
            # Merchant = standard merchant
            merchant = random.choice(merchant_list)
            is_new_merchant = 0 if random.random() < 0.85 else 1
            
            # Timestamp within active hours
            hour = random.randint(user["typical_hours_start"], user["typical_hours_end"])
            txn_timestamp = current_time.replace(hour=hour, minute=random.randint(0, 59))
            
        else:
            # Fraudulent transaction with layered domain anomalies
            anomaly_type = random.choice(["high_amount_new_device", "velocity_burst", "odd_hour_location_jump", "high_risk_merchant"])
            
            if anomaly_type == "high_amount_new_device":
                # Amount 6x to 25x normal mean
                amount = round(user["typical_amount_mean"] * random.uniform(6.0, 25.0) + random.uniform(5000, 45000), 2)
                device_id = f"DEV_UNKNOWN_{random.randint(100, 999)}"
                is_new_device = 1
                location = random.choice([c for c in CITIES if c != user["home_city"]])
                merchant = random.choice([m for m in merchant_list if m["category"] in ["Jewelry", "Electronics", "P2P"]])
                is_new_merchant = 1
                hour = random.choice([1, 2, 3, 4, 23])
                txn_timestamp = current_time.replace(hour=hour)
                v_1m += random.randint(2, 4)
                v_5m += random.randint(5, 9)
                v_60m += random.randint(8, 15)

            elif anomaly_type == "velocity_burst":
                amount = round(user["typical_amount_mean"] * random.uniform(2.0, 5.0), 2)
                device_id = device_by_user[u_id][0] if random.random() < 0.5 else f"DEV_UNKNOWN_{random.randint(100, 999)}"
                is_new_device = 1 if device_id != device_by_user[u_id][0] else 0
                location = user["home_city"]
                merchant = random.choice([m for m in merchant_list if m["category"] == "P2P"])
                is_new_merchant = 1
                hour = current_time.hour
                txn_timestamp = current_time
                v_1m += random.randint(3, 6)
                v_5m += random.randint(6, 12)
                v_60m += random.randint(10, 20)

            elif anomaly_type == "odd_hour_location_jump":
                amount = round(user["typical_amount_mean"] * random.uniform(4.0, 10.0), 2)
                device_id = f"DEV_UNKNOWN_{random.randint(100, 999)}"
                is_new_device = 1
                location = random.choice([c for c in CITIES if c != user["home_city"]])
                merchant = random.choice(merchant_list)
                is_new_merchant = 1
                hour = random.choice([2, 3, 4])
                txn_timestamp = current_time.replace(hour=hour)

            else:  # high_risk_merchant
                amount = round(random.uniform(15000, 85000), 2)
                device_id = device_by_user[u_id][0]
                is_new_device = 0
                location = user["home_city"]
                merchant = random.choice([m for m in merchant_list if m["category"] in ["Jewelry", "Electronics"]])
                is_new_merchant = 1
                hour = current_time.hour
                txn_timestamp = current_time

        # Update user timestamp history
        user_txn_timestamps[u_id].append(txn_timestamp)
        # Keep last 100 timestamps
        if len(user_txn_timestamps[u_id]) > 100:
            user_txn_timestamps[u_id].pop(0)

        # Compute derived telemetry features
        amount_z_score = round((amount - user["typical_amount_mean"]) / max(1.0, user["typical_amount_std"]), 2)
        location_deviation = 0 if location == user["home_city"] else 1
        
        # Hour deviation from user's active window
        if user["typical_hours_start"] <= hour <= user["typical_hours_end"]:
            hour_deviation = 0
        else:
            dist_before = abs(hour - user["typical_hours_start"])
            dist_after = abs(hour - user["typical_hours_end"])
            hour_deviation = min(dist_before, dist_after)

        # Generate Kaggle PCA vectors
        pca_vec = generate_kaggle_pca_vectors(1, is_fraud)

        row = {
            "txn_id": f"TXN_{idx:07d}",
            "user_id": u_id,
            "amount": amount,
            "merchant_id": merchant["id"],
            "merchant_name": merchant["name"],
            "merchant_category": merchant["category"],
            "device_id": device_id,
            "location": location,
            "timestamp": txn_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "hour_of_day": hour,
            "amount_z_score": amount_z_score,
            "is_new_device": is_new_device,
            "is_new_merchant": is_new_merchant,
            "location_deviation": location_deviation,
            "hour_deviation": hour_deviation,
            "velocity_1m": v_1m,
            "velocity_5m": v_5m,
            "velocity_60m": v_60m,
            "is_fraud": is_fraud
        }
        # Unpack PCA features
        row.update(pca_vec)
        transactions.append(row)

    return pd.DataFrame(transactions)

def main():
    print("Generating FinShield Hybrid UPI Fraud Dataset...")
    users, devices = generate_user_profiles(num_users=1000)
    merchants = generate_merchants()
    
    df_txns = generate_transactions(users, merchants, devices, total_txns=25000, fraud_ratio=0.035)
    
    # Save datasets
    df_txns.to_csv(os.path.join(DATA_DIR, "upi_transactions.csv"), index=False)
    with open(os.path.join(DATA_DIR, "users.json"), "w") as f:
        json.dump(users, f, indent=2)
    with open(os.path.join(DATA_DIR, "merchants.json"), "w") as f:
        json.dump(merchants, f, indent=2)
    with open(os.path.join(DATA_DIR, "devices.json"), "w") as f:
        json.dump(devices, f, indent=2)

    print(f"Dataset generated successfully!")
    print(f"Total Transactions: {len(df_txns)}")
    print(f"Legitimate Transactions: {(df_txns['is_fraud'] == 0).sum()}")
    print(f"Fraudulent Transactions: {(df_txns['is_fraud'] == 1).sum()} ({(df_txns['is_fraud'].mean()*100):.2f}%)")
    print(f"Saved files to {DATA_DIR}")

if __name__ == "__main__":
    main()
