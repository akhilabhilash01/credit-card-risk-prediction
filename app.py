import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.title("Credit Card Risk Prediction")

#Load models
models = {
    "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "kNN": joblib.load("model/knn.pkl"),
    "Gaussian Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl"),
    "XGBoost": joblib.load("model/xgboost.pkl")
}
scaler = joblib.load("model/scaler.pkl")

# Load features
with open("model/feature_columns.json") as f:
    feature_columns = json.load(f)
    
# Model selection
model_choice = st.selectbox("Select Model", list(models.keys()))

# Input features
st.subheader("Enter customer details to predict credit risk")

annual_income = st.number_input("Annual Income", value=50000)
children = st.number_input("Number of Children", value=0)
family_members = st.number_input("Family Members", value=2)
birthday_count = st.number_input("Birthday Count", value=12000)
employed_days = st.number_input("Employed Days", value=2000)

# Create empty row with all features
input_dict = dict.fromkeys(feature_columns, 0)

# Fill input value
if "Annual_income" in input_dict:
    input_dict["Annual_income"] = annual_income

if "CHILDREN" in input_dict:
    input_dict["CHILDREN"] = children

if "Family_Members" in input_dict:
    input_dict["Family_Members"] = family_members

if "Birthday_count" in input_dict:
    input_dict["Birthday_count"] = birthday_count

if "Employed_days" in input_dict:
    input_dict["Employed_days"] = employed_days

input_df = pd.DataFrame([input_dict])

# Prediction
if st.button("Predict"):
    model = models[model_choice]

    if model_choice in ["Logistic Regression", "KNN", "Naive Bayes"]:
        input_df = scaler.transform(input_df)

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("High Credit Risk")
    else:
        st.success("Low Credit Risk")

# Evaluation metrics
st.subheader("Model Evaluation Metrics")
try:
    metrics_df = pd.read_csv("results/comparison_table.csv")
    st.dataframe(metrics_df)
except:
    st.warning("Comparison table not found. Run training script first.")

# Confusion matrix
st.subheader("Confusion Matrix")
model_display_name = model_choice.replace(" ", "_")
cm_path = f"results/confusion_matrices/{model_display_name}_confusion_matrix.csv"
if os.path.exists(cm_path):
    cm_df = pd.read_csv(cm_path, index_col=0)

    fig, ax = plt.subplots()
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)
else:
    st.warning("Confusion matrix not found for this model.")


