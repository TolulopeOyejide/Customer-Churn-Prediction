# Customer Churn Prediction App

Customer Churn Prediction App is a comprehensive solution for predicting customer churn using a machine learning model, a FastAPI for serving predictions, and a Streamlit app for a user-friendly interface.



## The Demo Video

Here's a quick demo video that shows the Customer Churn Prediction App in action.

[**Watch the Demo Video**](https://www.youtube.com/watch?v=E49AZRhfk6w)



## Features
-   **Data Management**:
    -   Handles data cleaning and preprocessing.

-   **Automated Monitoring & Tracking**:
    -   Tracks experiments and manages model lineage with MLflow.

-   **Advanced Model Development**:
    -   Features an automated Grid Search CV pipeline for robust model training.
    -   Explores a comprehensive suite of logistic regression algorithms to select the best forming:
        -   Logistic Regression: A linear model used as a baseline for classification. A StandardScaler is applied to normalize the data before training.
        -   Decision Tree: A non-linear, tree-based model that captures hierarchical decision rules in the data.
        -   Random Forest Classifier: An ensemble method that builds multiple decision trees and merges their predictions to improve accuracy and control overfitting.
        -   Gradient Boosting Classifier: Another powerful ensemble method that builds trees sequentially, with each new tree correcting the errors of the previous one.
        - XGBoost: A highly optimized gradient boosting implementation known for its speed and performance.
        - K-Nearest Neighbors (KNN): A non-parametric, instance-based learning algorithm that classifies data points based on the majority class of their nearest neighbors. A StandardScaler is used to ensure all features contribute equally.


-   **API Service**:
    -   A FastAPI provides a RESTful API endpoint to serve real-time predictions.


-   **Web Application**:
    -   A Streamlit app offers a simple web interface for users to input data and receive predictions.


## Project Structure
```
Customer-Churn-Prediction/
│
├── data/                     # Raw and processed datasets
├── src/                      # Source code for the pipeline
│   ├── process_data.py       # Cleans and preprocesses data
│   ├── train_log_models.py   # Train and log model
│   └── load_model.py         # Supply the best trained and latest model to API
├── models/
│   ├── features/
│   │   └── training_features_YYYYMMDD_HHMMSS.pkl  
│   └── best_model_YYYYMMDD_HHMMSS.pkl            
├── app/                      # Source code for the application
│   ├── main.py               # Serving the model with Fast API
│   └── app.py                # Streamlit UI of the Customer Churn Prediction App
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation
```


## Setup Instructions
1. Clone the repository:  <br>
   `https://github.com/TolulopeOyejide/Customer-Churn-Prediction.git`



2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


3. Test the model API:  <br>
  `uvicorn app.main:app --host 0.0.0.0 --port 8005` <br>