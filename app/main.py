import os
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from fastapi.middleware.cors import CORSMiddleware

# Add the project's root directory to the Python path
# This allows you to import modules from `src/`
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_root)

from src.load_model import load_latest_model

#Initialize FastAPI app
app = FastAPI(title="Customer Churn Prediction API", description="An API to predict customer churn.", version="1.0.0")


#Add CORS Middleware after the app instance is created
app.add_middleware( CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],)



# Global State for Model and Features
api_state = { "model": None, "training_features": []}

def process_data_for_prediction(df: pd.DataFrame, training_features: list, numerical_features: list, categorical_features: list):
    """
    Preprocesses incoming data to match the training data format.
    This function is now self-contained within the API file.
    
    Args:
        df (pd.DataFrame): The raw input DataFrame.
        training_features (list): The list of features the model was trained on.
        numerical_features (list): The list of numerical features.
        categorical_features (list): The list of categorical features.
    
    Returns:
        pd.DataFrame: The preprocessed DataFrame ready for prediction.
    """
    # Normalize column names to lowercase and handle spaces
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # One-hot encode categorical features.
    df_processed = pd.get_dummies(df, columns=categorical_features)

    # Align columns with the training features.
    final_df = df_processed.reindex(columns=training_features, fill_value=0)
    
    # Drop any extra columns that were not in the training set
    return final_df.loc[:, training_features]


# Model Loading on Startup
@app.on_event("startup")
def load_model_on_startup():
    """Load the latest trained model and its associated features."""
    try:
        # Pass the correct model directory path, relative to the project root
        model_directory_path = os.path.join(project_root, "models")
        model, training_features = load_latest_model(model_directory_path)
        api_state["model"] = model
        api_state["training_features"] = training_features
        print("Model and features loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Pydantic Input Schema
class ChurnPredictionInput(BaseModel):
    age: int
    tenure: int
    usage_frequency: int
    support_calls: int
    payment_delay: int
    total_spend: float
    last_interaction: int
    gender: str
    subscription_type: str
    contract_length: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 35, "tenure": 12, "usage_frequency": 20, "support_calls": 2, 
                "payment_delay": 5, "total_spend": 120.50, "last_interaction": 2, 
                "gender": "Male", "subscription_type": "Premium", "contract_length": "Monthly"
            }
        }

# Prediction Endpoint
@app.post("/predict", tags=["Prediction"])
def predict_churn(input_data: ChurnPredictionInput):
    """
    Predicts the churn status of a customer based on input data.
    """
    if api_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not available.")

    try:
        df_raw = pd.DataFrame([input_data.dict()])
        numerical_features = ['age', 'tenure', 'usage_frequency', 'support_calls', 'payment_delay', 'total_spend', 'last_interaction']
        categorical_features = ['gender', 'subscription_type', 'contract_length']

        # Call the self-contained preprocessing function
        processed_df = process_data_for_prediction(
            df_raw, 
            api_state["training_features"],
            numerical_features,
            categorical_features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data preprocessing failed: {e}")

    try:
        prediction = api_state["model"].predict(processed_df)[0]
        probability = api_state["model"].predict_proba(processed_df)[0, 1]

        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "output_label": "Customer will churn" if prediction == 1 else "Customer will not churn"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
        

# Health Check Endpoint
@app.get("/", tags=["Health"])
def read_root():
    return {"status": "ok", "message": "Customer Churn Prediction API is running."}