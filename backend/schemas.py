from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class TransactionScoreRequest(BaseModel):
    user_id: str = Field(..., example="USR_0042")
    amount: float = Field(..., example=12500.00)
    merchant_id: Optional[str] = Field(default="MERCH_016", example="MERCH_016")
    device_id: str = Field(..., example="DEV_UNKNOWN_999")
    location: str = Field(..., example="Kolkata")
    timestamp: Optional[str] = Field(default=None, example="2026-09-02 02:45:00")

class RiskFactorResponse(BaseModel):
    factor: str
    contribution: float

class TransactionScoreResponse(BaseModel):
    transaction_id: str
    user_id: str
    amount: float
    location: str
    device_id: str
    merchant_id: Optional[str] = None
    risk_score: float
    decision: str
    top_factors: List[RiskFactorResponse]
    created_at: str

class TransactionListItem(BaseModel):
    id: str
    user_id: str
    amount: float
    merchant_name: Optional[str] = "Unknown Merchant"
    location: str
    device_id: str
    timestamp: str
    risk_score: float
    decision: str
    top_factors: List[RiskFactorResponse]

class StatsResponse(BaseModel):
    total_transactions: int
    flagged_count: int
    flagged_pct: float
    avg_risk_score: float
    model_metrics: dict

class SampleScenario(BaseModel):
    id: str
    title: str
    type: str  # "LEGIT", "STEP_UP", "FRAUD"
    description: str
    payload: TransactionScoreRequest
