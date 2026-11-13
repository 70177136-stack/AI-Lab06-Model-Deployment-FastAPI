# Lab 06: Machine Learning Model Deployment with FastAPI

**Author:** Mohsin Raza  
**SAP ID:** 70177136  
**Email:** 70177136@student.uol.edu.pk  
**Institution:** The University of Lahore (UOL)  
**Course:** Introduction to Artificial Intelligence (AI-101L)  

---

## 📌 Overview
This repository contains the full deployment code for **Laboratory 06**, packaging a machine learning diabetes prediction model into a high-performance RESTful microservice with a modern web user interface.

---

## 📁 Repository Structure
```
Lab06_Model_Deployment_FastAPI/
├── main.py              # FastAPI server implementation with /predict endpoint
├── index.html           # Interactive web dashboard with sample autofill
├── Mymdoel.pkl          # Serialized scikit-learn model artifact
├── model.pkl            # Standard convention alias copy of model artifact
├── test_api.py          # Automated integration test script
├── report.md            # Formal laboratory technical report
├── requirements.txt     # Python dependencies
└── README.md            # Execution manual
```

---

## 🚀 How to Run the Server

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the FastAPI Server
```bash
uvicorn main:app --reload --port 8000
```
The server will start at `http://localhost:8000`.

### 3. Access Interactive Endpoints
- **Interactive Web Interface:** Open `http://localhost:8000/app` or double-click `index.html` in your browser.
- **Swagger Interactive API Documentation:** Open `http://localhost:8000/docs`
- **ReDoc Documentation:** Open `http://localhost:8000/redoc`

### 4. Run Automated Test Suite
In a separate terminal:
```bash
python test_api.py
```
