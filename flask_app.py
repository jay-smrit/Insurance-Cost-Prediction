from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

#Load model
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    #Inputs
    Age = data.get("Age")
    Diabetes = data.get("Diabetes")
    BloodPressureProblems = data.get("BloodPressureProblems")
    AnyTransplants = data.get("AnyTransplants")
    AnyChronicDiseases = data.get("AnyChronicDiseases")
    Height = data.get("Height")
    Weight = data.get("Weight")
    KnownAllergies = data.get("KnownAllergies")
    HistoryOfCancerInFamily = data.get("HistoryOfCancerInFamily")
    NumberOfMajorSurgeries = data.get("NumberOfMajorSurgeries")

    input_df = pd.DataFrame([[Age, Diabetes, BloodPressureProblems, AnyTransplants, AnyChronicDiseases, Height, Weight, KnownAllergies, HistoryOfCancerInFamily, NumberOfMajorSurgeries]],
                            columns=['Age', 'Diabetes', 'BloodPressureProblems', 'AnyTransplants', 'AnyChronicDiseases', 'Height', 'Weight', 'KnownAllergies', 'HistoryOfCancerInFamily', 'NumberOfMajorSurgeries'])

    prediction = rf_model.predict(input_df)[0]

    return jsonify({"predicted_premium_price": prediction})

if __name__ == "__main__":
     app.run(debug=True)