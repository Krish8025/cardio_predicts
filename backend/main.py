from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Cardio Disease Prediction API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cardio_model_gb_optimized.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

model = None
scaler = None

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model/scaler: {e}")

class CardioInput(BaseModel):
    age: int  # in days
    gender: int # 1: women, 2: men
    height: int # cm
    weight: float # kg
    ap_hi: int # Systolic blood pressure
    ap_lo: int # Diastolic blood pressure
    cholesterol: int # 1: normal, 2: above normal, 3: well above normal
    gluc: int # 1: normal, 2: above normal, 3: well above normal
    smoke: int # 1 if smoke, 0 otherwise
    alco: int # 1 if alcohol, 0 otherwise
    active: int # 1 if active, 0 otherwise

@app.get("/")
def read_root():
    return {"message": "Welcome to Cardio Prediction API"}

@app.post("/predict")
def predict(data: CardioInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Prepare input dataframe
    input_dict = data.dict()
    
    # Feature Engineering
    # Calculate age_years and bmi
    input_dict['age_years'] = input_dict['age'] / 365.0
    input_dict['bmi'] = input_dict['weight'] / ((input_dict['height'] / 100) ** 2)
    
    # Domain-knowledge interaction features (must match training)
    # Smoke interactions
    input_dict['smoke_age'] = input_dict['smoke'] * input_dict['age_years']
    input_dict['smoke_bp'] = input_dict['smoke'] * input_dict['ap_hi']
    input_dict['smoke_chol'] = input_dict['smoke'] * input_dict['cholesterol']
    
    # Alcohol interactions
    input_dict['alco_age'] = input_dict['alco'] * input_dict['age_years']
    input_dict['alco_bp'] = input_dict['alco'] * input_dict['ap_hi']
    input_dict['alco_chol'] = input_dict['alco'] * input_dict['cholesterol']
    
    # Inactivity + BMI interaction
    input_dict['inactive_bmi'] = (1 - input_dict['active']) * input_dict['bmi']
    
    # Blood pressure ratio
    ap_lo_safe = input_dict['ap_lo'] if input_dict['ap_lo'] != 0 else 1
    input_dict['bp_ratio'] = input_dict['ap_hi'] / ap_lo_safe
    
    # Column order must match training exactly (20 features)
    ordered_cols = [
        'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
        'smoke', 'alco', 'active', 'age_years', 'bmi',
        'smoke_age', 'smoke_bp', 'smoke_chol',
        'alco_age', 'alco_bp', 'alco_chol',
        'inactive_bmi', 'bp_ratio'
    ]
    
    input_data = pd.DataFrame([input_dict])[ordered_cols]
    
    # Scale data
    try:
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)
        probability = model.predict_proba(scaled_data)[0][1] # Probability of class 1 (Cardio Disease)
        
        result = int(prediction[0])
        return {
            "prediction": result,
            "probability": float(probability),
            "status": "High Risk" if result == 1 else "Low Risk"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
