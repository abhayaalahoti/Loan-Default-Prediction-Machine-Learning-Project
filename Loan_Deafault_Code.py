# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 09:55:38 2025

@author: dell
"""
#=====================================================
# 1. IMPORT LIBRARIES
# =====================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)



def woe_iv_calculation(data, feature, target):
    
    #Calculate WOE & IV for a single feature
    

    df = data[[feature, target]].copy()

    # Total good & bad
    total_good = (df[target] == 0).sum()
    total_bad = (df[target] == 1).sum()

    # Group by feature
    grouped = df.groupby(feature)

    woe_df = grouped[target].agg(
        total='count',
        bad='sum'
    ).reset_index()

    # Good = Total - Bad
    woe_df['good'] = woe_df['total'] - woe_df['bad']

    # Distribution
    woe_df['dist_good'] = woe_df['good'] / total_good
    woe_df['dist_bad'] = woe_df['bad'] / total_bad

    # Handle zero division using smoothing
    woe_df['dist_good'] = woe_df['dist_good'].replace(0, 0.0001)
    woe_df['dist_bad'] = woe_df['dist_bad'].replace(0, 0.0001)

    # WOE calculation
    woe_df['WOE'] = np.log(woe_df['dist_good'] / woe_df['dist_bad'])

    # IV calculation
    woe_df['IV'] = (woe_df['dist_good'] - woe_df['dist_bad']) * woe_df['WOE']

    return woe_df, woe_df['IV'].sum()

# =====================================================
# 2. LOAD DATASET
# =====================================================

# Load Kaggle Telco Churn dataset
df = pd.read_csv("C:/Users/dell/Desktop/Loan_Default1.csv")

# =====================================================
# 3. BASIC DATA CLEANING
# =====================================================

features = df.columns.to_list()
features.remove('ID')
features.remove('Status')
print(features)

target = 'Status'

iv_summary = []

for feature in features:
    woe_table, iv_value = woe_iv_calculation(df, feature, target)

    print(f"\nWOE Table for: {feature}")
    print(woe_table)
    iv_summary.append({
        'Feature': feature,
        'IV': iv_value
    })

iv_df = pd.DataFrame(iv_summary).sort_values(by='IV', ascending=False)

print("\nInformation Value (IV) Summary:")
print(iv_df)

features_to_remove = iv_df.loc[
    (iv_df['IV'] <= 0.02) | (iv_df['IV'] >= 0.6),
    'Feature'
].tolist()

print(features_to_remove)

# drop these features from your main dataset
for i in features_to_remove:
    df.drop(i, axis=1, inplace=True)


customer_id = df['ID']

# Drop customerID as it has no predictive value
df.drop('ID', axis=1, inplace=True)

# =====================================================
# 4. FEATURE & TARGET SEPARATION
# =====================================================
def models(imp_df , output_df):
    
  X = df.drop('Status', axis=1)   # Features
  y = df['Status']               # Target

# =====================================================
# 5. ENCODE CATEGORICAL VARIABLES
# =====================================================

  le = LabelEncoder()

# Encode target variable (Yes=1, No=0)
  y = le.fit_transform(y)

# Encode all categorical features
  for col in X.columns:
      if X[col].dtype == 'object':
        X[col] = le.fit_transform(X[col])
        

# =====================================================
# 6. TRAIN TEST SPLIT
# =====================================================

  X_train, X_test, y_train, y_test, cust_train, cust_test = train_test_split(
      X, y, customer_id, test_size=0.3, random_state=42, stratify=y
  )



# =====================================================
# 7. DECISION TREE MODEL
# =====================================================

  dt_model = DecisionTreeClassifier(
      max_depth=8,             # Controls overfitting
      min_samples_leaf=40,     # Minimum samples in leaf node
     random_state=42
 )

# Train Decision Tree
   dt_model.fit(X_train, y_train)

# Predictions
  dt_pred = dt_model.predict(X_test)
  dt_prob = dt_model.predict_proba(X_test)[:, 1]

# =====================================================
# 8. RANDOM FOREST MODEL
# =====================================================

  rf_model = RandomForestClassifier(
     n_estimators=100,        # Number of trees
     max_depth=6,             # Controls tree depth
     min_samples_leaf=40,
     random_state=42,
     n_jobs=-1                # Use all CPU cores
  )

# Train Random Forest
  rf_model.fit(X_train, y_train)

# Predictions
  rf_pred = rf_model.predict(X_test)
  rf_prob = rf_model.predict_proba(X_test)[:, 1]

# =====================================================
# 9. MODEL EVALUATION FUNCTION
# =====================================================

  def evaluate_model(name, y_test, y_pred, y_prob):
     print(f"\n================ {name} ================")
     print("Accuracy:", accuracy_score(y_test, y_pred))
     print("ROC AUC:", roc_auc_score(y_test, y_prob))
     print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
     print("Classification Report:\n", classification_report(y_test, y_pred))

# Evaluate Decision Tree
  evaluate_model("DECISION TREE", y_test, dt_pred, dt_prob)

# Evaluate Random Forest
  evaluate_model("RANDOM FOREST", y_test, rf_pred, rf_prob)

# =====================================================
# 10. FEATURE IMPORTANCE COMPARISON
# =====================================================

  dt_importance = pd.DataFrame({
    'Feature': X.columns,
    'DecisionTree_Importance': dt_model.feature_importances_
  })

  rf_importance = pd.DataFrame({
    'Feature': X.columns,
    'RandomForest_Importance': rf_model.feature_importances_
  })

# Merge importance from both models
  importance_comparison = dt_importance.merge(
     rf_importance,
     on='Feature'
  ).sort_values(by='RandomForest_Importance', ascending=False)

  print("\nTop 10 Important Features (Comparison):")
  print(importance_comparison.head(10))

# =====================================================
# 11. FINAL COMPARISON SUMMARY
# =====================================================

  comparison_summary = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest'],
    'Accuracy': [
        accuracy_score(y_test, dt_pred),
        accuracy_score(y_test, rf_pred)
    ],
    'ROC_AUC': [
        roc_auc_score(y_test, dt_prob),
        roc_auc_score(y_test, rf_prob)
    ]
 })

 print("\nModel Performance Summary:")
 print(comparison_summary)


 output_df = pd.DataFrame({
    'customerID': cust_test.values,
    'Actual_Churn': y_test,
    'DecisionTree_Prediction': dt_pred,
    'DecisionTree_Probability': dt_prob,
    'RandomForest_Prediction': rf_pred,
    'RandomForest_Probability': rf_prob
 })


output_df=("C:/Users/dell/Desktop/Book19.xlsx")
imp_df=("C:/Users/dell/Desktop/Book20.xlsx")

print("SUCCESS: Excel exported")

#=======================================================================

    
features = [
    'property_value',
    'dtir1',
    'income',
    'lump_sum_payment',
    'loan_type',
    'neg_amortization',
    'co_applicant_credit_type',
    'submission_of_application',
    'business_or_commercial',
    'loan_amount'
]

remaining_features = features[:-3]
print("Remaining features:")
print(remaining_features)

features_to_remove = iv_df.loc[
    (iv_df['IV'] <= 0.1) | (iv_df['IV'] >= 0.6),
    'Feature'
].tolist()

print(features_to_remove)

# drop these features from your main dataset


customer_id = df['ID']

# Drop customerID as it has no predictive value
df.drop('ID', axis=1, inplace=True)

# =====================================================
# 4. FEATURE & TARGET SEPARATION
# =====================================================

X = df.drop('Status', axis=1)   # Features
y = df['Status']               # Target

# =====================================================
# 5. ENCODE CATEGORICAL VARIABLES
# =====================================================

le = LabelEncoder()

# Encode target variable (Yes=1, No=0)
y = le.fit_transform(y)

# Encode all categorical features
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = le.fit_transform(X[col])
        

# =====================================================
# 6. TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test, cust_train, cust_test = train_test_split(
    X, y, customer_id, test_size=0.3, random_state=42, stratify=y
)



# =====================================================
# 7. DECISION TREE MODEL
# =====================================================

dt_model = DecisionTreeClassifier(
    max_depth=8,             # Controls overfitting
    min_samples_leaf=40,     # Minimum samples in leaf node
    random_state=42
)

# Train Decision Tree
dt_model.fit(X_train, y_train)

# Predictions
dt_pred = dt_model.predict(X_test)
dt_prob = dt_model.predict_proba(X_test)[:, 1]

# =====================================================
# 8. RANDOM FOREST MODEL
# =====================================================

rf_model = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    max_depth=6,             # Controls tree depth
    min_samples_leaf=40,
    random_state=42,
    n_jobs=-1                # Use all CPU cores
)

# Train Random Forest
rf_model.fit(X_train, y_train)

# Predictions
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# =====================================================
# 9. MODEL EVALUATION FUNCTION
# =====================================================

def evaluate_model(name, y_test, y_pred, y_prob):
    print(f"\n================ {name} ================")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Evaluate Decision Tree
evaluate_model("DECISION TREE", y_test, dt_pred, dt_prob)

# Evaluate Random Forest
evaluate_model("RANDOM FOREST", y_test, rf_pred, rf_prob)

# =====================================================
# 10. FEATURE IMPORTANCE COMPARISON
# =====================================================

dt_importance = pd.DataFrame({
    'Feature': X.columns,
    'DecisionTree_Importance': dt_model.feature_importances_
})

rf_importance = pd.DataFrame({
    'Feature': X.columns,
    'RandomForest_Importance': rf_model.feature_importances_
})

# Merge importance from both models
importance_comparison = dt_importance.merge(
    rf_importance,
    on='Feature'
).sort_values(by='RandomForest_Importance', ascending=False)

print("\nTop 7 Important Features (Comparison):")
print(importance_comparison.head(7))

# =====================================================
# 11. FINAL COMPARISON SUMMARY
# =====================================================

comparison_summary = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest'],
    'Accuracy': [
        accuracy_score(y_test, dt_pred),
        accuracy_score(y_test, rf_pred)
    ],
    'ROC_AUC': [
        roc_auc_score(y_test, dt_prob),
        roc_auc_score(y_test, rf_prob)
    ]
})

print("\nModel Performance Summary:")
print(comparison_summary)


output_df = pd.DataFrame({
    'customerID': cust_test.values,
    'Actual_Churn': y_test,
    'DecisionTree_Prediction': dt_pred,
    'DecisionTree_Probability': dt_prob,
    'RandomForest_Prediction': rf_pred,
    'RandomForest_Probability': rf_prob
})


output_df.to_excel("C:/Users/dell/Desktop/Book20.xlsx", index=False)

print("SUCCESS: Excel exported")

















