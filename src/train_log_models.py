import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pickle
import glob
import time
from datetime import datetime
import warnings
import os
import joblib

# Import modules for classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)

# Import classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import process_data  # Import your preprocessing script

warnings.filterwarnings("ignore")

# Define paths and other constants
RAW_DATA_PATH = "data/raw/customer_data.csv"
CLEAN_DATA_DIR = "data/cleaned/"
MODEL_DIR = "models/"
FEATURES_SUBDIR = "features/"
EXPERIMENT_NAME = "Churn_Prediction"
TARGET_COL = "churn"

def train_and_log_model(
    data_source: str = "from_file", 
    X_train=None, X_test=None, y_train=None, y_test=None,
    model_dir: str = MODEL_DIR,
    experiment_name: str = EXPERIMENT_NAME,
    target_col: str = TARGET_COL):
    """
    Trains multiple classification models using GridSearchCV and MLflow tracking.
    Saves the best performing model to a pickle file.
    
    This function is designed to be flexible: it can either load data from a pre-processed
    file or accept pre-loaded data as arguments.

    Args:
        data_source (str): 'from_file' to load data from `process_data.py`, or 'from_args' to
                          use the provided X/y dataframes.
        X_train (pd.DataFrame, optional): Pre-loaded training features. Required if data_source='from_args'.
        X_test (pd.DataFrame, optional): Pre-loaded test features. Required if data_source='from_args'.
        y_train (pd.Series, optional): Pre-loaded training target. Required if data_source='from_args'.
        y_test (pd.Series, optional): Pre-loaded test target. Required if data_source='from_args'.
        model_dir (str): Directory to save the best model.
        experiment_name (str): The name for the MLflow experiment.
        target_col (str): The name of the target column for data loading.
    
    Returns:
        tuple: Path to the saved best model, name of the best model, and best model's ROC AUC score.
    """

    os.makedirs(model_dir, exist_ok=True)

    if data_source == "from_file":
        try:
            print("Loading and preprocessing data from file...")

            df_cleaned = process_data.clean_column_names(pd.read_csv(RAW_DATA_PATH, encoding="ISO-8859-1"))
            
            # Define feature lists based on the cleaned column names.
            numerical_features = [ 'age', 'tenure', 'usage_frequency', 'support_calls', 'payment_delay', 'total_spend', 'last_interaction']
            categorical_features = ['gender', 'subscription_type', 'contract_length']

            X_train, X_test, y_train, y_test = process_data.process_data_for_classification(df_cleaned, target_col=target_col,
             numerical_features=numerical_features,categorical_features=categorical_features)

        except Exception as e:
            print(f"Error loading or preprocessing data: {e}")
            return None, None, None
    
    # Verify that the data is loaded correctly
    if X_train is None or X_test is None or y_train is None or y_test is None:
        print("Data is not properly loaded. Exiting.")
        return None, None, None

    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    print(X_train.info())  # Correctly prints info
    print(X_test.info())  # Correctly prints info
    
    models = {
        "LogisticRegression": (Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(random_state=42))]),{"classifier__penalty": ["l1", "l2"], "classifier__C": [0.1, 1.0, 10.0]}),
        "DecisionTree": (DecisionTreeClassifier(random_state=42),{"max_depth": [5, 10, None], "min_samples_split": [2, 5]}),
        "RandomForest": (RandomForestClassifier(random_state=42),{"n_estimators": [100, 200], "max_depth": [5, 10, None]}),
        "GradientBoosting": (GradientBoostingClassifier(random_state=42),{"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]}),
        "XGBoost": (XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss"),{"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.01, 0.1]}),
        "KNN": (Pipeline([('scaler', StandardScaler()), ('classifier', KNeighborsClassifier())]),{"classifier__n_neighbors": [3, 5, 7]}),
        #"SVC": (Pipeline([('scaler', StandardScaler()), ('classifier', SVC(random_state=42, probability=True))]),{"classifier__C": [0.1, 1, 10], "classifier__kernel": ["linear", "rbf"]}),
        #"MLPClassifier": (Pipeline([('scaler', StandardScaler()), ('classifier', MLPClassifier(random_state=42, max_iter=500))]),{"classifier__hidden_layer_sizes": [(50,), (100,)], "classifier__alpha": [0.0001, 0.001]})
    }

    best_model, best_name, best_score = None, None, -1.0
    mlflow.set_experiment(experiment_name)

    # Training Loop
    for name, (model, params) in models.items():
        if mlflow.active_run():
            mlflow.end_run()
        print(f"\n--- Training {name} ---")
        start_time = time.time()

        scoring_metric = "roc_auc"
        if not hasattr(model, 'predict_proba') and name not in ["SVC", "LogisticRegression", "KNN", "MLPClassifier"]:
            scoring_metric = "f1"
            print(f"Warning: {name} does not support `predict_proba`. Using 'f1' for scoring.")
        
        try:
            grid = GridSearchCV(model, params, cv=5, scoring=scoring_metric, n_jobs=-1, verbose=1)
            
            with mlflow.start_run(run_name=name):
                grid.fit(X_train, y_train)
                elapsed_time = time.time() - start_time

                y_pred = grid.best_estimator_.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                roc_auc = None
                if hasattr(grid.best_estimator_, 'predict_proba'):
                    y_pred_proba = grid.best_estimator_.predict_proba(X_test)[:, 1]
                    roc_auc = roc_auc_score(y_test, y_pred_proba)

                mlflow.log_params(grid.best_params_)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                
                if roc_auc is not None:
                    mlflow.log_metric("roc_auc", roc_auc)
                    print(f"{name} -> ROC AUC: {roc_auc:.4f}, F1: {f1:.4f}, Training Time: {elapsed_time:.2f} sec")
                else:
                    print(f"{name} -> F1: {f1:.4f}, Training Time: {elapsed_time:.2f} sec")

                mlflow.sklearn.log_model(grid.best_estimator_, name)

                if roc_auc is not None and roc_auc > best_score:
                    best_score = roc_auc
                    best_model = grid.best_estimator_
                    best_name = name
                elif roc_auc is None and f1 > best_score:
                    best_score = f1
                    best_model = grid.best_estimator_
                    best_name = name

        except Exception as e:
            print(f"An error occurred during training {name}: {e}")
            continue

    features_dir = os.path.join(model_dir, FEATURES_SUBDIR)
    os.makedirs(features_dir, exist_ok=True)

    if best_model is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        best_model_path = os.path.join(model_dir, f"best_model_{timestamp}.pkl")

        with open(best_model_path, "wb") as f:
            pickle.dump(best_model, f)

        features_path = os.path.join(features_dir, f"training_features_{timestamp}.pkl")
        with open(features_path, "wb") as f:
            joblib.dump(list(X_train.columns), f)

        print(f"\nBest Model: {best_name} with Best Score: {best_score:.4f}")
        print(f"Saved to {best_model_path}")
        print(f"Saved training features to {features_path}")
        return best_model_path, best_name, best_score
    else:
        print("\nNo models were successfully trained. Aborting model saving.")
        return None, None, None


if __name__ == "__main__":
    train_and_log_model()