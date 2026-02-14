import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Credit Card Risk Prediction")
st.write("Enter customer details to predict credit risk.")

#Load models
models = {
    "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "KNN": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl"),
    "XGBoost": joblib.load("model/xgboost.pkl")
}
scaler = joblib.load("model/scaler.pkl")

# Model selection
model_choice = st.selectbox("Select Model", list(models.keys()))

# Input features
st.subheader("Enter Sample Numeric Features")

annual_income = st.number_input("Annual Income", value=50000)
children = st.number_input("Number of Children", value=0)
family_members = st.number_input("Family Members", value=2)
birthday_count = st.number_input("Birthday Count", value=12000)
employed_days = st.number_input("Employed Days", value=2000)

# Create input array
input_data = np.array([[annual_income, children, family_members, birthday_count, employed_days]])

# Scale input
if model_choice in ["Logistic Regression", "KNN", "Naive Bayes"]:
    input_data = scaler.transform(input_data)

# Predeiction
if st.button("Predict"):
    model = models[model_choice]
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("High Credit Risk")
    else:
        st.success("Low Credit Risk")