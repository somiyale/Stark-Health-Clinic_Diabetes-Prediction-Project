model = joblib.load("diabetes_model.joblib")

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
HBA1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, step=0.1)
glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=500)

# Preprocessing
gender_enc = gender_map[gender]
smoking_enc = smoking_history_map[smoking]
hypertension_enc = yes_no_map[hypertension]
heart_disease_enc = yes_no_map[heart_disease]

# Feature engineering
age_bmi_interaction = age * bmi
glucose_hba1c_interaction = glucose * hba1c

# Age group encoding
age_group_adult = age >= 18 and age < 50
age_group_senior = age >= 50 and age < 65
age_group_elder = age >= 65

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
    "HbA1c_level": hba1c,
    "blood_glucose_level": glucose,
    "age_bmi_interaction": age_bmi_interaction,
    "glucose_hba1c_interaction": glucose_hba1c_interaction,
    "age_group_Adult": age_group_adult,
    "age_group_Senior": age_group_senior,
    "age_group_Elder": age_group_elder,
    "bmi_category_Obese": bmi_obese,
    "bmi_category_Overweight": bmi_overweight,
    "bmi_category_Underweight": bmi_underweight
}])

# Making Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f" Likely to have diabetes (Probability: {proba:.2f})")
    else:
        st.success(f" Unlikely to have diabetes (Probability: {proba:.2f})")
