# diabetes_app.py
import streamlit as st
import pandas as pd
from joblib import load


# Load model
model = load("diabetes_model.joblib")


# App interface
st.title("Diabetes Prediction System")

# Input fields
st.header("Patient Details")
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.slider("Pregnancies", 0, 17, 3)
    glucose = st.slider("Glucose", 0, 200, 117)
    bp = st.slider("Blood Pressure", 0, 122, 72)
    skin = st.slider("Skin Thickness", 0, 99, 23)

with col2:
    insulin = st.slider("Insulin", 0, 846, 30)
    bmi = st.slider("BMI", 0.0, 67.1, 32.0)
    dpf = st.slider("Diabetes Pedigree", 0.078, 2.42, 0.3725)
    age = st.slider("Age", 21, 81, 29)

# Prediction
if st.button("Predict"):
    input_data = [[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]]
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]
    
    st.subheader("Result")
    if prediction == 1:
        st.error(f"Diabetic (Probability: {proba:.1%})")
    else:
        st.success(f"Not Diabetic (Probability: {proba:.1%})")

# Model info
st.sidebar.header("Model Information")
st.sidebar.write("Algorithm: Random Forest")
st.sidebar.write("Parameters: 100 trees")
