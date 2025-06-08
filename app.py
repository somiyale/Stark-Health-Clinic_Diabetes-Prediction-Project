import joblib
import streamlit as st
import pandas as pd

model = joblib.load('diabetes_model.joblib')

# Simple mappings (must match what was used in training)
gender_map = {"Male": 0, "Female": 1}
smoking_history_map = {"Never": 0, "Former": 1, "Current": 2, "Not Current": 3, "No Info": 4}
yes_no_map = {"No": 0, "Yes": 1}

# # Streamlit U
st.title("Diabetes Prediction App")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=120, step=1)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1)
smoking = st.selectbox("Smoking History", ["Never", "Former", "Current", "Not Current", "No Info"])
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
HBA1c = st.number_input("HBA1c Level", min_value=3.0, max_value=15.0, step=0.1)
glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=500)

# Preprocessing
gender_enc = gender_map[gender]
smoking_enc = smoking_history_map[smoking]
hypertension_enc = yes_no_map[hypertension]
heart_disease_enc = yes_no_map[heart_disease]

# Feature engineering
age_bmi_interaction = age * bmi
glucose_hba1c_interaction = glucose * HBA1c


# Age group encoding
age_group = pd.cut([age], bins=[0, 30, 50, 70, 100], labels=['Young', 'Adult', 'Senior', 'Elder'])[0]

age_group_Adult = int(age_group == 'Adult')
age_group_Senior = int(age_group == 'Senior')
age_group_Elder = int(age_group == 'Elder')

# BMI category encoding
bmi_obese = bmi >= 30
bmi_overweight = bmi >= 25 and bmi < 30
bmi_underweight = bmi < 18.5

# Assemble features into a DataFrame
input_data = pd.DataFrame([{
    "gender": gender_enc,
    "age": age,
    "hypertension": hypertension_enc,
    "heart_disease": heart_disease_enc,
    "smoking_history": smoking_enc,
    "bmi": bmi,
    "HbA1c_level": HBA1c,
    "blood_glucose_level": glucose,
    "age_bmi_interaction": age_bmi_interaction,
    "glucose_hba1c_interaction": glucose_hba1c_interaction,
    "age_group_Adult": age_group_Adult,
    "age_group_Senior": age_group_Senior,
    "age_group_Elder": age_group_Elder,
    "bmi_category_Obese": bmi_obese,
    "bmi_category_Overweight": bmi_overweight,
    "bmi_category_Underweight": bmi_underweight
}])

# Ensuring all expected model columns are present
for col in model.feature_names_in_:
    if col not in input_data.columns:
        input_data[col] = 0

# Ensuring column order matches model
input_data = input_data[model.feature_names_in_]


# Making Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f" Likely to have diabetes (Probability: {proba:.2f})")
    else:
        st.success(f" Unlikely to have diabetes (Probability: {proba:.2f})")
