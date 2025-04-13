import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

# App title
st.title("Customer Churn Prediction")
st.write("Enter the customer information below to predict whether they will churn.")

# Input fields for features
gender = st.selectbox("Gender", ['Male', 'Female'])
partner = st.selectbox("Partner", ['Yes', 'No'])
dependents = st.selectbox("Dependents", ['Yes', 'No'])
tenure = st.number_input("Tenure (in months)", min_value=0, max_value=72)
phone_service = st.selectbox("Phone Service", ['Yes', 'No'])
internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
online_security = st.selectbox("Online Security", ['Yes', 'No'])
online_backup = st.selectbox("Online Backup", ['Yes', 'No'])
device_protection = st.selectbox("Device Protection", ['Yes', 'No'])
tech_support = st.selectbox("Tech Support", ['Yes', 'No'])
streaming_tv = st.selectbox("Streaming TV", ['Yes', 'No'])
streaming_movies = st.selectbox("Streaming Movies", ['Yes', 'No'])
contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
payment_method = st.selectbox("Payment Method", ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

# Convert inputs to the format used by the model
input_data = pd.DataFrame([[gender, partner, dependents, tenure, phone_service, internet_service,
                            online_security, online_backup, device_protection, tech_support, streaming_tv,
                            streaming_movies, contract, payment_method, monthly_charges, total_charges]],
                          columns=['gender', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'InternetService',
                                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                                   'StreamingMovies', 'Contract', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'])

# Encode the input data
for column in ['gender', 'Partner', 'Dependents', 'PhoneService', 'InternetService', 'OnlineSecurity',
               'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
               'Contract', 'PaymentMethod']:
    input_data[column] = label_encoders[column].transform(input_data[column])

# Scale the features
input_data_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_data_scaled)

# Show prediction
if prediction[0] == 1:
    st.write("This customer is likely to **churn**.")
else:
    st.write("This customer is **not likely to churn**.")
