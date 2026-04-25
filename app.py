import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Load trained model & pipeline
# -------------------------------
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Insurance Premium Predictor", layout="centered")

st.title("💰 Insurance Premium Prediction App")
st.markdown("Enter customer details to estimate the insurance premium.")

# -------------------------------
# Input Section
# -------------------------------
st.header("📋 Customer Information")

# Continuous Inputs
age = st.slider("Age", 18, 100, 30)
height = st.slider("Height (cm)", 100, 220, 170)
weight = st.slider("Weight (kg)", 30, 150, 70)

# Categorical Inputs (Binary)
def binary_input(label):
    return st.selectbox(label, ["No", "Yes"])

diabetes = binary_input("Diabetes")
bp = binary_input("Blood Pressure Problems")
transplants = binary_input("Any Transplants")
chronic = binary_input("Any Chronic Diseases")
allergies = binary_input("Known Allergies")
cancer_history = binary_input("History of Cancer in Family")

# Numeric discrete
surgeries = st.number_input("Number of Major Surgeries", min_value=0, max_value=10, step=1)

# -------------------------------
# Convert Inputs
# -------------------------------
def encode_binary(value):
    return 1 if value == "Yes" else 0

input_data = pd.DataFrame({
    "Age": [age],
    "Diabetes": [encode_binary(diabetes)],
    "BloodPressureProblems": [encode_binary(bp)],
    "AnyTransplants": [encode_binary(transplants)],
    "AnyChronicDiseases": [encode_binary(chronic)],
    "Height": [height],
    "Weight": [weight],
    "KnownAllergies": [encode_binary(allergies)],
    "HistoryOfCancerInFamily": [encode_binary(cancer_history)],
    "NumberOfMajorSurgeries": [surgeries]
})

# -------------------------------
# Prediction
# -------------------------------
st.markdown("---")

if st.button("🔍 Predict Premium"):
    try:
        prediction = rf_model.predict(input_data)[0]

        st.success(f"💸 Estimated Premium: ₹ {prediction:,.2f}")

        # Optional interpretation
        if prediction < 10000:
            st.info("Low Risk Profile")
        elif prediction < 30000:
            st.warning("Moderate Risk Profile")
        else:
            st.error("High Risk Profile")

    except Exception as e:
        st.error(f"Error: {e}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit")