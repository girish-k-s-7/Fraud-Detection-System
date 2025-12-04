# 🚨 Fraud Detection System (End-to-End ML Pipeline + Deployment)

This project implements a **production-ready Fraud Detection System** using Machine Learning with a complete pipeline including data ingestion, data transformation, model training, evaluation, and deployment using Streamlit.

The model predicts whether a credit card transaction is **Fraud / Not Fraud** based on transaction features. The project follows industry-grade software engineering practices and modular design.

---

## 🧠 Problem Statement

Financial fraud causes massive losses to businesses every year.  
Detecting fraudulent transactions early helps reduce risk and protect users.

This project aims to build a **supervised machine learning system** that accurately classifies fraudulent transactions using historical transaction data.

---

## 📁 Dataset Overview

The dataset contains anonymized transaction features and a binary fraud label.

### 🔹 Dataset Columns

| Column Name | Description |
|--------------|-------------|
| V1–V28 | Anonymized transaction features |
| Amount | Transaction amount |
| Time | Time since first transaction |
| Class | Target variable (0 = Legitimate, 1 = Fraud) |

---

## 🎯 Target Variable

### `Class`
Indicates whether the transaction is fraudulent.

- 0 → Legitimate  
- 1 → Fraud  

The dataset is **highly imbalanced**, making Recall and ROC-AUC critical evaluation metrics.

---

## ⚙️ Tech Stack

| Layer | Technologies |
|--------|-------------|
| Programming | Python |
| ML Models | Logistic Regression, Random Forest |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Scaling | RobustScaler |
| Evaluation | Precision, Recall, ROC-AUC |
| Deployment | Streamlit |
| Serialization | Joblib |
| Logging | Custom Logger |
| Error Handling | Custom Exceptions |
| Version Control | Git, GitHub |

---

## 🔄 ML Pipeline Workflow

### ✅ Data Ingestion
Reads dataset and splits data into training and testing sets.

### ✅ Data Transformation
Performs:

- Feature scaling using RobustScaler
- Separation of target variable
- Saves preprocessing object (`preprocessor.pkl`)

### ✅ Model Training
Two models were trained and evaluated:

- Logistic Regression (baseline)
- Random Forest (final model)

---

## 📊 Model Comparison

| Model | Recall | Precision | ROC-AUC |
|--------|--------|-----------|---------|
| Logistic Regression | High | Very Low | Very High |
| Random Forest | Balanced | High | Best |

---

## 🏆 Best Model

✅ **Random Forest Classifier**

Reason:
- Best ROC-AUC score
- High precision
- Reasonable recall
- Handles imbalance effectively

---

## 🌐 Streamlit Web Application

The application allows:

- Selecting transactions
- Running predictions
- Displaying probability
- Viewing ground truth

### ✅ Prediction Output:
- Fraud / Not Fraud
- Fraud probability score

---

---

## 🏆 Key Highlights

✅ Modular Architecture  
✅ Clean Logging  
✅ Custom Exceptions  
✅ End-to-end Pipeline  
✅ Imbalance-aware metrics  
✅ Model comparison  
✅ Production deployment  
✅ Interview-ready system design  

---

## 🚀 Future Improvements

- SHAP explainability
- Threshold tuning
- CI/CD pipeline
- Docker deployment
- Cloud hosting
- Drift detection

---

## 👨‍💻 Author

**Girish K S**  
Data Scientist 

---

