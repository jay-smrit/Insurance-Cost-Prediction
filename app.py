import streamlit as st
import pandas as pd
import pickle

with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

st.title("Insurance Premium Calculator")

st.write("Please enter insurer details below")

Age = st.number_input("Age", min_value=18, max_value=150, value=30)
Diabetes = st.selectbox("Diabetes", options=[0, 1])
BloodPressureProblems = st.selectbox("BloodPressure Problems", options=[0, 1])
AnyTransplants = st.selectbox("Any Transplants", options=[0, 1])
AnyChronicDiseases = st.selectbox("Any Chronic Diseases", options=[0, 1])
Height = st.number_input("Height (in cm)", min_value=0, max_value=250, value=170)
Weight = st.number_input("Weight (in kg)", min_value=0, max_value=300, value=70)
KnownAllergies = st.selectbox("Known Allergies", options=[0, 1])
HistoryOfCancerInFamily = st.selectbox("History Of Cancer In Family", options=[0, 1])
NumberOfMajorSurgeries = st.selectbox("NumberOfMajorSurgeries", options=[0, 1,2,3])

input_df = pd.DataFrame([[Age, Diabetes, BloodPressureProblems, AnyTransplants, AnyChronicDiseases, Height, Weight, KnownAllergies, HistoryOfCancerInFamily, NumberOfMajorSurgeries]],
                        columns=['Age', 'Diabetes', 'BloodPressureProblems', 'AnyTransplants', 'AnyChronicDiseases', 'Height', 'Weight', 'KnownAllergies', 'HistoryOfCancerInFamily', 'NumberOfMajorSurgeries'])

st.dataframe(input_df)

if st.button("Predict Premium Price"):
    prediction = rf_model.predict(input_df)[0]
    st.write(f"The predicted premium price is:  {prediction:,.2f}")

