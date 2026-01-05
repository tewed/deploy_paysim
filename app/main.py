from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np
import os
import xgboost
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# define input data schema
class TransactionData(BaseModel):

    type:  object
    amount: float
    oldbalanceOrig: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

    class Config:
        schema_extra = {
            "example": 
            {
                "type": "CASH_OUT",
                "amount": 230.0,
                "oldbalanceOrg": 26000.05,
                "newbalanceOrig": 123456.00,
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

def load_artifacts():
    '''Load model and artifacts'''
    model = pickle.load(open("./pickles/fraud_model.pkl", "rb"))
    scaler = pickle.load(open("./pickles/scaler.pkl", "rb"))
    ohe = pickle.load(open("./pickles/onehot_encoder.pkl", "rb"))
    return model, scaler, ohe
 
# create our prediction endpoint:
@app.post("/predict")
def predict_fraud(transaction: TransactionData):
    # Load model and preprocessing objects
    model, scaler, ohe = load_artifacts()
    input_df = pd.DataFrame([{ 
                'amount':transaction.amount,
                'oldbalanceOrg':transaction.oldbalanceOrg,
                'newbalanceOrig': transaction.newbalanceOrig,
                'oldbalanceDest':transaction.oldbalanceDest,
                'newbalanceDest':transaction.newbalanceDest,
                'type':transaction.type,
                    }
                ])
    # Convert fields to model input
    df_ohe = ohe.fit_transform(input_df[['type']])

    train_ohe_df =  pd.DataFrame(df_ohe, columns=ohe.get_feature_names_out())
    # drop original type column
    df = pd.concat([df.drop('type', axis=1), train_ohe_df], axis=1)
    
    df = df.replace([float('inf'), float('-inf')], 0).fillna(0)
    X_scaled = scaler.fit_transform(df)

    # Make prediction
    prediction = model.predict(X_scaled)[0]

    # Return result with additional context
    return {
        "predicted_progression_score": round(prediction, 2),
        "interpretation": get_interpretation(prediction)
    }

# Create an interpretation function to allow for human readable interpretation of scores
def get_interpretation(score):
    """Create interpretation of the prediction score"""
    if score <= 0.5:
        return "Not-Fraud"
    else:
        return "Fraud"

@app.get("/")
def health_check():
    return {"status": "healthy", "model": "paysim_reduced_v1"}



