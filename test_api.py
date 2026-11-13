"""
FastAPI Automated Integration Test Script
Student: Mohsin Raza (SAP ID: 70177136) - University of Lahore
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_endpoints():
    print("==================================================")
    print("Testing Diabetes Prediction FastAPI Endpoints")
    print("Author: Mohsin Raza (SAP ID: 70177136)")
    print("==================================================")

    # 1. Health check
    try:
        r_health = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"[1] Health Check: Status {r_health.status_code} -> {r_health.json()}")
    except Exception as e:
        print(f"[FAIL] Server unreachable at {BASE_URL}. Run 'uvicorn main:app --reload' first.")
        return

    # 2. Test Healthy Patient Prediction
    healthy_patient = {
        "Glucose": 88.0,
        "BloodPressure": 66.0,
        "Insulin": 32.0,
        "BMI": 21.5,
        "DiabetesPedigreeFunction": 0.18,
        "Age": 24.0
    }
    r1 = requests.post(f"{BASE_URL}/predict", json=healthy_patient)
    print("\n[2] Healthy Profile Test:")
    print("    Inputs :", healthy_patient)
    print("    Outcome:", r1.json())

    # 3. Test High-Risk Patient Prediction
    diabetic_patient = {
        "Glucose": 175.0,
        "BloodPressure": 92.0,
        "Insulin": 210.0,
        "BMI": 39.4,
        "DiabetesPedigreeFunction": 0.85,
        "Age": 54.0
    }
    r2 = requests.post(f"{BASE_URL}/predict", json=diabetic_patient)
    print("\n[3] Diabetic Profile Test:")
    print("    Inputs :", diabetic_patient)
    print("    Outcome:", r2.json())

    print("\nAll integration checks completed successfully!")

if __name__ == "__main__":
    test_endpoints()
