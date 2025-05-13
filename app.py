import streamlit as st
import numpy as np
# import pickle
import joblib

# with open("model.pkl", "rb") as f:
#     model = pickle.load(f)

model = joblib.load("model.pkl")

st.set_page_config(page_title="Alzheimer's Prediction", layout="centered")
st.title("🧠 Alzheimer's Disease Prediction using Machine Learning")

st.markdown("""
Please fill in the details below to predict the chance of Alzheimer's Disease.
""")

age = st.number_input("Age", min_value=0, max_value=100, value=50)
alcohol = st.selectbox("Alcohol Consumption", ["Yes", "No"])
family_history = st.selectbox("Family History of Alzheimer's", ["Yes", "No"])
hypertension = st.selectbox("Hypertension", ["Yes", "No"])
mmse = st.slider("MMSE Score", 0, 30, value=15)
functional_assessment = st.slider("Functional Assessment Score", 0, 10, value=5)
memory_complaints = st.selectbox("Memory Complaints", ["Yes", "No"])
behavioral_problems = st.selectbox("Behavioral Problems", ["Yes", "No"])
adl = st.slider("Activities of Daily Living (ADL)", 0, 10, value=5)
disorientation = st.selectbox("Disorientation Present", ["Yes", "No"])

def to_binary(value):
    return 1 if value == "Yes" else 0

input_features = np.array([[
    age,
    to_binary(alcohol),
    to_binary(family_history),
    to_binary(hypertension),
    mmse,
    functional_assessment,
    to_binary(memory_complaints),
    to_binary(behavioral_problems),
    adl,
    to_binary(disorientation)
]])

if st.button("🔍 Predict Diagnosis"):
    try:
        prediction = model.predict(input_features)[0]

        if prediction == 1:
            st.error("🧠 Diagnosis: **POSITIVE for Alzheimer's Disease**")
        else:
            st.success("🧠 Diagnosis: **NEGATIVE for Alzheimer's Disease**")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_features)[0]
            st.info(f"🧪 Prediction Confidence: {max(proba) * 100:.2f}%")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")


