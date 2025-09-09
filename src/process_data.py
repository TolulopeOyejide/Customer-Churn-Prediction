import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib
import pickle
import os
from datetime import datetime

# Define constants
RAW_DATA_PATH = "data/raw/customer_data.csv"
CLEAN_DATA_DIR = "data/cleaned/"

# Create cleaned folder if it doesn't exist
os.makedirs(CLEAN_DATA_DIR, exist_ok=True)


# ---- Statutory Data Quality Checks & Fixes ----

def clean_column_names(df):
    """Make column headers lowercase and remove whitespace for consistency."""
    df.columns = (
        df.columns.str.strip()  # remove leading/trailing spaces
        .str.lower()  # make lowercase
        .str.replace(" ", "_")  # replace spaces with underscore
    )
    return df


def check_missing_values(df):
    """Returns a DataFrame with counts and percentages of missing values per column."""
    missing = df.isnull().sum()
    percent = (missing / len(df)) * 100
    return pd.DataFrame({'missing_count': missing, 'missing_percent': percent}).sort_values(by='missing_count', ascending=False)



def handle_missing_values(df, strategy="mean", fill_value=None):
    """Handle missing values based on chosen strategy."""
    if strategy == "mean":
        return df.fillna(df.mean(numeric_only=True))
    elif strategy == "median":
        return df.fillna(df.median(numeric_only=True))
    elif strategy == "mode":
        return df.fillna(df.mode().iloc[0])
    elif strategy == "drop":
        return df.dropna()
    elif strategy == "constant":
        return df.fillna(fill_value)
    else:
        raise ValueError("Invalid strategy. Choose from: mean, median, mode, drop, constant.")

        
def remove_duplicates(df):
    """Remove duplicate rows."""
    return df.drop_duplicates()


def handle_outliers(df, column=None):
    """Clip outliers in a column using IQR."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.clip(df[column], lower_bound, upper_bound)
    return df


def check_class_balance(df, target):
    """Returns the distribution of classes in the target column."""
    return df[target].value_counts(normalize=True) * 100


def handle_class_imbalance(X, y, method="smote"):
    """Handle class imbalance using different resampling techniques."""
    if method == "smote":
        sampler = SMOTE(random_state=42)
    elif method == "oversample":
        sampler = RandomOverSampler(random_state=42)
    elif method == "undersample":
        sampler = RandomUnderSampler(random_state=42)
    else:
        raise ValueError("method must be one of ['smote', 'oversample', 'undersample']")

    X_res, y_res = sampler.fit_resample(X, y)
    
    return X_res, y_res





def process_data_for_prediction(df: pd.DataFrame, training_features: list, 
                                numerical_features: list, categorical_features: list):
    """
    Preprocesses incoming data to match the training data format.
    
    Args:
        df (pd.DataFrame): The raw input DataFrame.
        training_features (list): The list of features the model was trained on.
        numerical_features (list): The list of numerical features.
        categorical_features (list): The list of categorical features.
    
    Returns:
        pd.DataFrame: The preprocessed DataFrame ready for prediction.
    """
    # This function uses a correct approach that avoids the 'unhashable type: list' error.

    # 1. Ensure all columns are lowercase and formatted correctly
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # 2. One-hot encode categorical features
    # This uses the correct pandas function which does not require a hashable type
    df_processed = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    # 3. Reindex to match the training features, filling missing columns with 0
    # The 'training_features' list is used to align columns, which is a correct use case
    final_df = df_processed.reindex(columns=training_features, fill_value=0)

    # 4. Return the processed DataFrame
    return final_df[training_features]







# ---- Main Preprocessing Pipeline ----

def process_data_for_classification(df, target_col='churn', numerical_features=None, categorical_features=None, test_size=0.2, random_state=42, 
scaling_method='standard',imbalance_method='smote'):

    """
    Performs a full data preprocessing pipeline for machine learning models.
    """
    print("--- Starting the Preprocessing Pipeline ---")

    # Step 1: Handle missing values and duplicates on the whole dataset
    df = handle_missing_values(df, strategy="mean")
    df = remove_duplicates(df)

    # Step 2: Separate features and target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the DataFrame.")
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Validate feature lists
    if numerical_features is None or categorical_features is None:
        raise ValueError("Please provide a list of numerical and categorical features.")

    missing_features = [f for f in numerical_features + categorical_features if f not in X.columns]
    if missing_features:
        raise ValueError(f"The following features are missing from the dataframe: {missing_features}")

    # Step 3: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print("\n--- Data Split into Training and Testing Sets ---")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Step 4: Preprocessing on Training Data Only to prevent data leakage
    print("\n--- Applying Preprocessing on Training Data ---")

    # Outlier Handling
    for col in numerical_features:
        X_train = handle_outliers(X_train, column=col)

    # One-hot encode categorical features
    X_train_encoded = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)

    # Step 5: Align columns after one-hot encoding
    print("\n--- Aligning columns for consistency ---")
    train_cols = X_train_encoded.columns
    test_cols = X_test_encoded.columns

    missing_in_test = list(set(train_cols) - set(test_cols))
    for col in missing_in_test:
        X_test_encoded[col] = 0
    X_test_encoded = X_test_encoded[train_cols]

    # Step 6: Handle Class Imbalance on training data only
    print("\n--- Handling Class Imbalance on Training Data ---")
    class_distribution_train = check_class_balance(df=pd.DataFrame(y_train), target=y_train.name)
    print("Class distribution in original training data:\n", class_distribution_train)

    X_train_resampled, y_train_resampled = handle_class_imbalance(X_train_encoded, y_train, method=imbalance_method)

    resampled_distribution = check_class_balance(df=pd.DataFrame(y_train_resampled), target=y_train_resampled.name)
    print("\nClass distribution after resampling:\n", resampled_distribution)

    print("\n--- Data Preprocessing Complete ---")
    
    return X_train_resampled, X_test_encoded, y_train_resampled, y_test


def save_processed_data(X_train, X_test, y_train, y_test, output_dir="data/cleaned/"):
    """Saves the preprocessed data to disk."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    X_train.to_csv(os.path.join(output_dir, f"X_train_{timestamp}.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, f"X_test_{timestamp}.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, f"y_train_{timestamp}.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, f"y_test_{timestamp}.csv"), index=False)
    print(f"Processed data saved to {output_dir}")


if __name__ == "__main__":
    try:
        # Load the raw data
        df_raw = pd.read_csv(RAW_DATA_PATH, encoding="ISO-8859-1")
        print(f"Raw Data Shape: {df_raw.shape}")

        # Clean the column names first to ensure consistency
        df_cleaned = clean_column_names(df_raw.copy())
        
        # Verify the columns after cleaning
        print("Cleaned DataFrame columns:", df_cleaned.columns.tolist())

        numerical_features = ['age', 'tenure', 'usage_frequency', 'support_calls', 'payment_delay', 'total_spend', 'last_interaction'] #adjustable

        categorical_features = ['gender', 'subscription_type', 'contract_length'] #adjustable
        
        # Now, call the function with the cleaned dataframe and feature lists
        X_train_resampled, X_test_encoded, y_train_resampled, y_test = process_data_for_classification(df=df_cleaned, target_col='churn', numerical_features=numerical_features,
         categorical_features=categorical_features)

        save_processed_data(X_train_resampled, X_test_encoded, y_train_resampled, y_test)

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure '{RAW_DATA_PATH}' exists.")
    except Exception as e:
        print(f"An error occurred during pipeline execution: {e}")