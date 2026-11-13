"""
FastAPI Model Deployment: Diabetes Prediction Service
Course: Introduction to Artificial Intelligence (AI-101L)
Author: Mohsin Raza (SAP ID: 70177136)
Email: 70177136@student.uol.edu.pk
Institution: The University of Lahore (UOL)
"""

import pickle
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="Clinical Diabetes Prediction API",
    description="Production Machine Learning inference endpoint built by Mohsin Raza (70177136) - University of Lahore",
    version="1.0.0"
)

# Enable Cross-Origin Resource Sharing (CORS) for web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Robust Model Loading (supports both Mymdoel.pkl and model.pkl)
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "Mymdoel.pkl"
if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR / "model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Extract expected feature ordering from trained estimator
try:
    FEATURES = list(model.feature_names_in_)
except AttributeError:
    FEATURES = ["Glucose", "BloodPressure", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]


class PredictRequest(BaseModel):
    Glucose: float = Field(..., example=148.0, description="Plasma glucose concentration (mg/dL)")
    BloodPressure: float = Field(..., example=72.0, description="Diastolic blood pressure (mm Hg)")
    Insulin: float = Field(..., example=125.0, description="2-Hour serum insulin (mu U/ml)")
    BMI: float = Field(..., example=33.6, description="Body mass index (weight in kg/(height in m)^2)")
    DiabetesPedigreeFunction: float = Field(..., example=0.627, description="Diabetes pedigree genetic score")
    Age: float = Field(..., example=50.0, description="Age in years")


@app.get("/", summary="Root Status & Info")
def home():
    """Returns service metadata and author details."""
    return {
        "service": "Clinical Diabetes Prediction API",
        "author": "Mohsin Raza",
        "sap_id": "70177136",
        "university": "The University of Lahore (UOL)",
        "status": "Online",
        "endpoints": {
            "web_ui": "/app",
            "documentation": "/docs",
            "health_check": "/health",
            "prediction": "/predict"
        }
    }


@app.get("/health", summary="Health Check")
def health():
    """Confirms inference service readiness."""
    return {"status": "ok", "model_loaded": True, "features": FEATURES}


@app.get("/app", summary="Web Interface")
def serve_frontend():
    """Serves the standalone HTML web user interface."""
    html_file = BASE_DIR / "index.html"
    if html_file.exists():
        return FileResponse(html_file)
    raise HTTPException(status_code=404, detail="index.html not found")


@app.post("/predict", summary="Predict Diabetes Risk")
def predict(req: PredictRequest):
    """
    Accepts patient biometric vitals and returns the binary classification
    along with estimated posterior probability and qualitative risk tier.
    """
    try:
        # Construct DataFrame ensuring strict column alignment
        input_data = pd.DataFrame([req.model_dump()])[FEATURES]
        
        pred_label = int(model.predict(input_data)[0])
        pred_prob = float(model.predict_proba(input_data)[0][1])
        
        # Determine qualitative risk level
        if pred_prob >= 0.70:
            risk = "High"
        elif pred_prob >= 0.40:
            risk = "Moderate"
        else:
            risk = "Low"
            
        return {
            "predicted_label": pred_label,
            "prediction": "Diabetic" if pred_label == 1 else "Not Diabetic",
            "probability": round(pred_prob, 3),
            "probability_percentage": f"{pred_prob * 100:.1f}%",
            "risk_level": risk,
            "patient_vitals": req.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("Starting Diabetes Prediction API server on http://localhost:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
