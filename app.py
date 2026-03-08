from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, create_model
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Insurance Fraud Detection API")

# Load model on startup
MODEL_PATH = "fraud_model.joblib"
try:
    model_pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model_pipeline = None

# Create arbitrary input schema based on our dataset features
# To keep UI simple, we'll request the most critical fields and dummy the rest, 
# or request a comprehensive list. For a real pipeline, we provide all.
class ClaimData(BaseModel):
    months_as_customer: int
    age: int
    policy_state: str
    policy_csl: str
    policy_deductable: int
    policy_annual_premium: float
    umbrella_limit: int
    insured_sex: str
    insured_education_level: str
    insured_occupation: str
    insured_hobbies: str
    insured_relationship: str
    capital_gains: int
    capital_loss: int
    incident_type: str
    collision_type: str
    incident_severity: str
    authorities_contacted: str
    incident_state: str
    incident_city: str
    incident_hour_of_the_day: int
    number_of_vehicles_involved: int
    property_damage: str
    bodily_injuries: int
    witnesses: int
    police_report_available: str
    total_claim_amount: float
    injury_claim: float
    property_claim: float
    vehicle_claim: float
    auto_make: str
    auto_year: int

# Mount static directory for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.post("/predict")
def predict_fraud(claim: ClaimData):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
        
    try:
        # Convert pydantic model to dataframe
        # Note the python model uses hyphenated names for capital gains/loss
        data_dict = claim.dict()
        data_dict['capital-gains'] = data_dict.pop('capital_gains')
        data_dict['capital-loss'] = data_dict.pop('capital_loss')
        
        input_df = pd.DataFrame([data_dict])
        
        # Predict
        prediction = model_pipeline.predict(input_df)
        probability = model_pipeline.predict_proba(input_df)[0][1] # Probability of Class 1 (Fraud)
        
        result = "Fraud" if prediction[0] == 1 else "Genuine"
        
        return {
            "prediction": result,
            "probability_of_fraud": float(probability),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
