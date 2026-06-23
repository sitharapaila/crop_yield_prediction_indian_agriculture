import streamlit as st
import pickle
import numpy as np

# Page settings
st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾")

# Load model
model = pickle.load(open("crop_model.pkl", "rb"))

st.title("🌾 Crop Yield Prediction System")
st.write("Enter details below to predict crop yield")

# Inputs
rainfall = st.number_input("Rainfall (mm)", value=800.0)
temperature = st.number_input("Temperature (°C)", value=28.0)
area = st.number_input("Area (hectares)", value=100.0)

# Predict
if st.button("Predict Yield"):
    features = np.array([[rainfall, temperature, area]])
    prediction = model.predict(features)

    st.success(f"🌾 Predicted Yield: {prediction[0]:.2f} tons/hectare")