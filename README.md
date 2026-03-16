# Loan-Default-Prediction-Machine-Learning-Project
Loan Default Prediction using Decision Tree and Random Forest classifiers with comparative performance analysis.

# Loan Default Prediction

A machine learning project that predicts whether a customer will default 
on a loan using supervised classification algorithms.

# Overview
This project applies and compares two ML models — Decision Tree and 
Random Forest — on a dataset of 44,601 customer records to identify 
high-risk borrowers.

# Model Performance
| Model           | Precision | Recall | Accuracy | F1 Score |
|----------------|-----------|--------|----------|----------|
| Decision Tree  |    93%    |  56%   |   88%    |   70%    |
| Random Forest  |    98%    |  47%   |   12%    |   64%    |

# Tech Stack
- Python
- Scikit-learn
- Pandas & NumPy
- Microsoft Excel (analysis & reporting)

# Key Findings
- Decision Tree achieved the best overall balance (F1: 70%, Accuracy: 88%)
- Random Forest showed high precision but suffered from class imbalance
- Dataset has ~1:5.8 default to non-default ratio
