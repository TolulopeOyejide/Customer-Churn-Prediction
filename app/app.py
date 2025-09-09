import streamlit as st
import pandas as pd
import requests

# Set the title and a brief description for the app
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="centered")

# Page Header
st.title("Customer Churn Prediction App")
st.write("""
This app predicts whether a customer will churn based on their demographic and usage data.
Fill in the form below to get a prediction from the machine learning model.
""")

# API Configuration
# You can change the API_URL if you deploy your API to a different server
# For local testing, use http://localhost:8005
API_URL = "http://localhost:8006/predict" 

# Input Form 
# The columns are defined here to match your FastAPI Pydantic schema
# We use st.columns to create a clean, two-column layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35, placeholder="e.g., 35")
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12, placeholder="e.g., 12")
    usage_frequency = st.number_input("Usage Frequency (per month)", min_value=0, max_value=100, value=20, placeholder="e.g., 20")
    support_calls = st.number_input("Support Calls", min_value=0, max_value=20, value=2, placeholder="e.g., 2")
    payment_delay = st.number_input("Payment Delay (days)", min_value=0, max_value=90, value=5, placeholder="e.g., 5")

with col2:
    total_spend = st.number_input("Total Spend ($)", min_value=0.0, value=120.50, format="%.2f", placeholder="e.g., 120.50")
    last_interaction = st.number_input("Last Interaction (days ago)", min_value=0, max_value=365, value=2, placeholder="e.g., 2")
    gender = st.selectbox("Gender", ["Male", "Female"], index=None, placeholder="Select Gender")
    subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"], index=None, placeholder="Select Subscription Type")
    contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annually"], index=None, placeholder="Select Contract Length")

# Prediction Button and Logic
if st.button("Predict Churn"):
    # Create the JSON payload from the form inputs
    payload = {
        "age": age,
        "tenure": tenure,
        "usage_frequency": usage_frequency,
        "support_calls": support_calls,
        "payment_delay": payment_delay,
        "total_spend": total_spend,
        "last_interaction": last_interaction,
        "gender": gender,
        "subscription_type": subscription_type,
        "contract_length": contract_length
    }
    
    # Check if any required field is empty
    if any(value is None for value in payload.values()):
        st.warning("Please fill in all the fields before predicting.")
    else:
        # Send a POST request to the FastAPI endpoint
        with st.spinner('Predicting...'):
            try:
                response = requests.post(API_URL, json=payload)
                
                # Check for successful response
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display the prediction result
                    st.subheader("Prediction Result")
                    st.metric(label=f"Predicted Churn: {result['output_label']}", 
                              value=f"{result['probability']:.2f}",
                              help="Probability of churn")

                    if result['prediction'] == 1:
                        st.error("Prediction: This customer is likely to churn.")
                    else:
                        st.success("Prediction: This customer is not likely to churn.")
                    
                    st.write(f"The model predicted a churn probability of **{result['probability']:.2f}**.")
                
                # Handle API errors
                else:
                    st.error(f"Error from API: {response.status_code}")
                    st.json(response.json())
            
            except requests.exceptions.ConnectionError:
                st.error("API connection failed. Please ensure your FastAPI server is running.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# Footer
st.markdown("<hr><p style='text-align:center; color:gray;'>Developed by Tolulope Oyejide AI/ML Labs</p>",unsafe_allow_html=True)