from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    home_city = Column(String, nullable=False)
    typical_amount_mean = Column(Float, default=500.0)
    typical_amount_std = Column(Float, default=200.0)
    typical_hours_start = Column(Integer, default=8)
    typical_hours_end = Column(Integer, default=22)

    devices = relationship("Device", back_populates="user", cascade="all, delete-orphan")
    transactions = relationship("Transaction", back_populates="user")

class Device(Base):
    __tablename__ = "devices"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    device_fingerprint = Column(String, nullable=False)
    first_seen_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="devices")

class Merchant(Base):
    __tablename__ = "merchants"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    category = Column(String, nullable=False)
    risk_weight = Column(Float, default=0.1)

    transactions = relationship("Transaction", back_populates="merchant")

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    amount = Column(Float, nullable=False)
    merchant_id = Column(String, ForeignKey("merchants.id"), nullable=True)
    merchant_name = Column(String, nullable=True)
    device_id = Column(String, nullable=False)
    location = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    risk_score = Column(Float, nullable=False)
    decision = Column(String, nullable=False)  # ALLOW, STEP_UP_AUTH, BLOCK
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="transactions")
    merchant = relationship("Merchant", back_populates="transactions")
    risk_factors = relationship("RiskFactor", back_populates="transaction", cascade="all, delete-orphan")

class RiskFactor(Base):
    __tablename__ = "risk_factors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String, ForeignKey("transactions.id"), nullable=False)
    factor_name = Column(String, nullable=False)
    contribution_pct = Column(Float, nullable=False)

    transaction = relationship("Transaction", back_populates="risk_factors")
