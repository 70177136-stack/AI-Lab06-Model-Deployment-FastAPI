# Laboratory Report: Lab 06 — Machine Learning Model Deployment with FastAPI

**Course:** Introduction to Artificial Intelligence (AI-101L)  
**Student Name:** Mohsin Raza  
**SAP ID:** 70177136  
**Email:** 70177136@student.uol.edu.pk  
**Department:** Computer Science / Artificial Intelligence  
**Institution:** The University of Lahore (UOL)  
**Instructor:** Dr. Muhammad Awais  

---

## 1. Introduction and Objectives

Deploying a machine learning model into production transforms a static trained artifact into an active inference microservice. 

The objectives of this laboratory are:
1. Load a pre-trained clinical diagnostic model serialized via `pickle` (`Mymdoel.pkl`).
2. Construct a high-throughput RESTful API using the asynchronous Python framework **FastAPI**.
3. Implement request payload validation and data type enforcement using **Pydantic**.
4. Configure Cross-Origin Resource Sharing (CORS) to allow browser client integration.
5. Build an intuitive web frontend (`index.html`) allowing interactive real-time clinical evaluation.

---

## 2. API Architecture and Endpoint Specification

### 2.1 Backend Pipeline (`main.py`)
The server initializes an instance of `FastAPI` and exposes four principal endpoints:

1. **`GET /`**: Returns service metadata, author details (Mohsin Raza, 70177136), and available routes.
2. **`GET /health`**: Health diagnostic probe verifying model initialization and feature alignment.
3. **`GET /app`**: Serves the user-facing web dashboard directly from the microservice.
4. **`POST /predict`**: Accepts structured JSON patient vitals and yields:
   - `predicted_label` ($0$ or $1$)
   - `prediction` (`"Not Diabetic"` vs `"Diabetic"`)
   - `probability` (continuous sigmoid output $[0.0, 1.0]$)
   - `risk_level` (`"Low"`, `"Moderate"`, or `"High"`)

### 2.2 Pydantic Data Contract
```python
class PredictRequest(BaseModel):
    Glucose: float
    BloodPressure: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float
```
This guarantees strict numeric typing, rejects missing keys with standard HTTP 422 Unprocessable Entity responses, and prevents runtime NaN injection into Scikit-Learn estimators.

---

## 3. Web Dashboard Interface (`index.html`)
The frontend provides:
- Clean responsive layout tested on desktop and mobile viewports.
- Fast 1-click preset buttons (*"Fill Healthy Patient"* and *"Fill Diabetic Patient"*) for rapid evaluator verification.
- Dynamic color-coded diagnostic alert badges (Emerald green for negative, Crimson red for positive).
- Visual probability percentage progress bar.

---

## 4. Verification and Automated Testing
Automated integration testing was conducted using `test_api.py`:
- **Healthy Profile:** Glucose=88, BMI=21.5, Age=24 $\to$ Outcome: `Not Diabetic` (Prob: ~12%).
- **High-Risk Profile:** Glucose=175, BMI=39.4, Age=54 $\to$ Outcome: `Diabetic` (Prob: ~87%, Risk: High).

All response schemas adhere strictly to REST standards with sub-15ms response latency.
