import os
import sys
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

# Add project root and backend dir to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
for path in [PROJECT_ROOT, BACKEND_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from ml.feature_engineering import compute_realtime_features
from ml.explainer import FinShieldExplainer
import models
import schemas

# Global singleton explainer instance
_explainer_instance = None

def get_explainer():
    global _explainer_instance
    if _explainer_instance is None:
        _explainer_instance = FinShieldExplainer()
    return _explainer_instance

def score_transaction_pipeline(db: Session, request: schemas.TransactionScoreRequest) -> dict:
    # 1. Fetch User Profile
    user = db.query(models.User).filter(models.User.id == request.user_id).first()
    if not user:
        # Create default fallback user profile if requested user ID not yet seeded
        user = models.User(
            id=request.user_id,
            name=f"User {request.user_id}",
            home_city=request.location,
            typical_amount_mean=500.0,
            typical_amount_std=200.0,
            typical_hours_start=8,
            typical_hours_end=22
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    # 2. Fetch user's registered devices and known merchants
    user_devices = [d.id for d in db.query(models.Device).filter(models.Device.user_id == user.id).all()]
    if not user_devices:
        user_devices = [f"DEV_{user.id}_01"]

    # 3. Query recent transaction timestamps for velocity calculation
    now = datetime.utcnow()
    one_hour_ago = now - timedelta(hours=1)
    recent_txns = db.query(models.Transaction.timestamp)\
                    .filter(models.Transaction.user_id == user.id, models.Transaction.timestamp >= one_hour_ago)\
                    .all()
    recent_timestamps = [t[0] for t in recent_txns]

    # 4. Fetch Merchant details
    merchant_name = "Unknown Merchant"
    if request.merchant_id:
        m = db.query(models.Merchant).filter(models.Merchant.id == request.merchant_id).first()
        if m:
            merchant_name = m.name

    # 5. Extract Feature Vector
    user_profile_dict = {
        "typical_amount_mean": user.typical_amount_mean,
        "typical_amount_std": user.typical_amount_std,
        "home_city": user.home_city,
        "typical_hours_start": user.typical_hours_start,
        "typical_hours_end": user.typical_hours_end,
        "known_device_ids": user_devices,
        "known_merchant_ids": [request.merchant_id] if request.merchant_id else []
    }

    txn_payload_dict = {
        "amount": request.amount,
        "location": request.location,
        "device_id": request.device_id,
        "merchant_id": request.merchant_id,
        "timestamp": request.timestamp or now.strftime("%Y-%m-%d %H:%M:%S")
    }

    df_single = compute_realtime_features(txn_payload_dict, user_profile_dict, recent_timestamps)

    # 6. Run Model Inference & SHAP Explanation
    explainer = get_explainer()
    result = explainer.score_and_explain(df_single)

    # 7. Generate Transaction ID & Save to Database
    txn_count = db.query(models.Transaction).count()
    new_txn_id = f"TXN_{txn_count + 1:07d}"

    dt_obj = datetime.strptime(txn_payload_dict["timestamp"], "%Y-%m-%d %H:%M:%S") if isinstance(txn_payload_dict["timestamp"], str) else now

    db_txn = models.Transaction(
        id=new_txn_id,
        user_id=user.id,
        amount=request.amount,
        merchant_id=request.merchant_id,
        merchant_name=merchant_name,
        device_id=request.device_id,
        location=request.location,
        timestamp=dt_obj,
        risk_score=result["risk_score"],
        decision=result["decision"],
        created_at=now
    )
    db.add(db_txn)

    # Save risk factors
    for factor in result["top_factors"]:
        rf = models.RiskFactor(
            transaction_id=new_txn_id,
            factor_name=factor["factor"],
            contribution_pct=factor["contribution"]
        )
        db.add(rf)

    db.commit()

    return {
        "transaction_id": new_txn_id,
        "user_id": user.id,
        "amount": request.amount,
        "location": request.location,
        "device_id": request.device_id,
        "merchant_id": request.merchant_id,
        "merchant_name": merchant_name,
        "risk_score": result["risk_score"],
        "decision": result["decision"],
        "top_factors": result["top_factors"],
        "created_at": now.strftime("%Y-%m-%d %H:%M:%S")
    }
