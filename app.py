import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

# Dummy model for demonstration (replace with your actual trained model)
model = LinearRegression()
# Assume your model is trained here...

# Create Streamlit interface
st.title("Insurance Premium Predictor")
age = st.number_input("Enter Age", min_value=0, max_value=100)
bmi = st.number_input("Enter BMI", min_value=0.0, max_value=50.0)
children = st.number_input("Enter Number of Children", min_value=0, max_value=10)

if st.button("Predict"):
    prediction = model.predict(np.array([[age, bmi, children]]))[0]
    st.success(f"Predicted Insurance Premium: ${prediction:.2f}")
