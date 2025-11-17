from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import math

app = FastAPI(title="Fraud Detection System")

# --- Load Artifacts ---
try:
    model = joblib.load("app/model.pkl")
    scaler = joblib.load("app/scaler.pkl")
    device_lookup = joblib.load("app/device_lookup.pkl")
    ip_lookup = joblib.load("app/ip_lookup.pkl")
    occ_lookup = joblib.load("app/occ_lookup.pkl")

    print("Artifacts loaded successfully.")
except Exception as e:
    print(f"CRITICAL: Could not load model artifacts. {e}")


class TransactionRequest(BaseModel):
    TransactionAmount: float = Field(..., description="Transaction value in USD")
    LoginAttempts: int = Field(
        ..., description="Number of failed logins before purchase"
    )
    DeviceID: str = Field(..., description="Unique device identifier")
    IP_Address: str = Field(..., description="IPv4 Address")
    AccountBalance: float = Field(
        ..., description="Account balance at time of transaction"
    )
    CustomerOccupation: str = Field(
        ..., description="Occupation (e.g., Doctor, Student, Retired, Engineer)"
    )
    Channel: str = Field(..., description="Transaction Channel (Online, ATM, Branch)")

    class Config:
        json_schema_extra = {
            "example": {
                "AccountBalance": 4000.00,
                "Channel": "Online",
                "DeviceID": "D000123",
                "IP_Address": "192.168.1.5",
                "LoginAttempts": 2,
                "CustomerOccupation": "Doctor",
                "TransactionAmount": 1500.50,
            }
        }


class FraudResponse(BaseModel):
    prediction: str
    risk_factors: list[str]
    fraud_probability: float


# --- Helper ---
def calculate_risk_prob(decision_score):
    prob = 1 / (1 + math.exp(10 * decision_score))
    return round(prob, 4)


# --- The Endpoint ---
@app.post("/predict", response_model=FraudResponse)
def predict_fraud(tx: TransactionRequest):
    risk_factors = []

    # 1. Feature Lookup
    unique_accounts_device = device_lookup.get(tx.DeviceID, 1)
    unique_accounts_ip = ip_lookup.get(tx.IP_Address, 1)
    occ_stats = occ_lookup.get(tx.CustomerOccupation, {"OccMean": 5000, "OccStd": 2000})

    if occ_stats["OccStd"] > 0:
        balance_zscore = (tx.AccountBalance - occ_stats["OccMean"]) / occ_stats[
            "OccStd"
        ]
    else:
        balance_zscore = 0

    if tx.AccountBalance > 0:
        amount_to_balance = tx.TransactionAmount / tx.AccountBalance
    else:
        amount_to_balance = 0

    # 2. Rule Engine (Context-Aware)

    if tx.Channel == "Online":
        if unique_accounts_device > 3:
            return FraudResponse(
                prediction="FRAUD",
                risk_factors=[
                    f"Device Farm Detected (Online Device linked to {unique_accounts_device} accounts)"
                ],
                fraud_probability=0.99,
            )
        # We also apply strict IP checks to Online channels
        if unique_accounts_ip > 3:
            return FraudResponse(
                prediction="FRAUD",
                risk_factors=[
                    f"Botnet IP Detected (Online IP linked to {unique_accounts_ip} accounts)"
                ],
                fraud_probability=0.99,
            )

    # 3. ML Model
    features_df = pd.DataFrame(
        [
            {
                "TransactionAmount": tx.TransactionAmount,
                "LoginAttempts": tx.LoginAttempts,
                "AmounttoBalanceRatio": amount_to_balance,
                "BalanceOccZScore": balance_zscore,
            }
        ]
    )
    try:
        features_scaled = scaler.transform(features_df)
        ml_score = model.decision_function(features_scaled)[0]
        risk_prob = calculate_risk_prob(ml_score)

        if risk_prob > 0.5:
            risk_factors.append("Anomalous Transaction Pattern (ML Detection)")

            if abs(balance_zscore) > 3:
                risk_factors.append(
                    f"Suspicious Balance for Occupation '{tx.CustomerOccupation}'"
                )

            return FraudResponse(
                prediction="FRAUD",
                risk_factors=risk_factors,
                fraud_probability=risk_prob,
            )
        else:
            return FraudResponse(
                prediction="NORMAL",
                risk_factors=["Transaction looks clean"],
                fraud_probability=risk_prob,
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model Inference Failed: {str(e)}")
