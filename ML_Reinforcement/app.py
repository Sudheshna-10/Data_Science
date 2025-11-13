import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load('model/model.pkl')

# Streamlit page settings
st.set_page_config(page_title="Heart Disease Prediction 🫀", layout="centered")

st.title("🫀 Heart Disease Prediction App")
st.write("This app predicts whether a person is likely to have heart disease based on medical attributes.")

# Input fields
st.header("Enter Patient Details:")

age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex", options=["Female", "Male"])
cp = st.selectbox("Chest Pain Type (cp)", options=["ASY", "ATA", "NAP", "TA"])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol (chol)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1])
restecg = st.selectbox("Resting ECG Results (restecg)", options=["LVH", "Normal", "ST"])
thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina (exang)", options=["No", "Yes"])
oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment (slope)", options=["Down", "Flat", "Up"])

# Encoding the categorical variables (must match training encodings)
def encode_inputs(sex, exang, restecg, slope, cp):
    # Encoding must match your LabelEncoder order in training
    sex_map = {"Female": 0, "Male": 1}
    exang_map = {"No": 0, "Yes": 1}
    restecg_map = {"LVH": 0, "Normal": 1, "ST": 2}
    slope_map = {"Down": 0, "Flat": 1, "Up": 2}
    cp_map = {"ASY": 0, "ATA": 1, "NAP": 2, "TA": 3}
    return sex_map[sex], exang_map[exang], restecg_map[restecg], slope_map[slope], cp_map[cp]

sex_encoded, exang_encoded, restecg_encoded, slope_encoded, cp_encoded = encode_inputs(sex, exang, restecg, slope, cp)

# Predict Button
if st.button("Predict"):
    # Create feature array
    features = np.array([[age, sex_encoded, cp_encoded, trestbps, chol, fbs,
                          restecg_encoded, thalach, exang_encoded, oldpeak, slope_encoded]])

    # Make prediction
    prediction = model.predict(features)[0]

    # Display result
    if prediction == 1:
        st.error("⚠️ The person is likely to have Heart Disease.")
    else:
        st.success(" The person is unlikely to have Heart Disease.")
