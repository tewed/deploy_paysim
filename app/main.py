from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os

# define input data schema
class TransactionData(BaseModel):
    step:                int32
    type:               object
    amount:            float32
    nameOrig:           object
    oldbalanceOrig:    float32
    newbalanceOrig:    float32
    nameDest:           object
    oldbalanceDest:    float32
    newbalanceDest:    float32
    # isFraud:              int8
    # isFlaggedFraud:       int8

    class Config:
        schema_extra = {
            "example": 
            {
                "step": 3,
                "type":"CASH_OUT",
                "amount": 230.0,
                "nameOrig":"C123456",
                "oldbalanceOrig": 26000.05,
                "newbalanceOrig": 123456.00,
                "nameDest": "M123456",
                "oldbalanceDest": 123.56,
                "newbalanceDest": 250.06
            }
        }

# initialize FastAPI app and load the model into the FastAPI environment


# Initialize FastAPI app
app = FastAPI(
    title="Paysim Fraud predictor",
    description="Predicts fraudulent transactions from PaySim data",
    version="1.0.0"
)
 
# Load the trained model
model_path = os.path.join("models", "paysim_model.pkl")
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# create our prediction endpoint:


