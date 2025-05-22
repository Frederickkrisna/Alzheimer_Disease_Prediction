import streamlit as st
import numpy as np
# import pickle
import joblib
import requests
from streamlit_lottie import st_lottie

# with open("model.pkl", "rb") as f:
#     model = pickle.load(f)

model = joblib.load("model.pkl")

def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_brain = load_lottie_url("https://lottie.host/16de5156-ec3e-4760-adf2-540b64ee45f5/U6b3IfJ1m1.json")

st.set_page_config(page_title="Alzheimer's Prediction", layout="centered", page_icon="🧠")

st_lottie(lottie_brain, speed=1, width=300, height=200, key="brain")

st.title("🧠 Alzheimer's Disease Prediction using Machine Learning")

st.markdown("""
Please fill in the details below to predict the chance of Alzheimer's Disease.
""")

st.markdown("---")

age = st.number_input("Age: The age of the patients ranges from 60 to 90 years.", min_value=60, max_value=90, value=75)
alcohol = st.slider("Alcohol Consumption: Weekly alcohol consumption in units.", 0, 20, value=14)
family_history = st.selectbox("Family History of Alzheimer's: Family history of Alzheimer's Disease.", ["Yes", "No"])
hypertension = st.selectbox("Hypertension: Presence of hypertension (high blood pressure).", ["Yes", "No"])
mmse = st.slider("MMSE Score: Mini-Mental State Examination score. Lower scores indicate cognitive impairment.", 0, 30, value=25)
functional_assessment = st.slider("Functional Assessment Score: Functional assessment score. Lower scores indicate greater impairment.", 0, 10, value=9)
memory_complaints = st.selectbox("Memory Complaints: Presence of memory complaints.", ["Yes", "No"])
behavioral_problems = st.selectbox("Behavioral Problems: Presence of behavioral problems.", ["Yes", "No"])
adl = st.slider("ADL Score: Activities of Daily Living score. Lower scores indicate greater impairment.", 0, 10, value=8)
disorientation = st.selectbox("Disorientation Present: Presence of disorientation", ["Yes", "No"])

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

st.markdown("----")

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
        