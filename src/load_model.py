import os
import glob
import joblib

def load_latest_model(model_dir: str):
    """
    Loads the most recently saved model and features files, regardless of timestamp match.
    
    WARNING: This approach can lead to a mismatch between the loaded model and the
    features it was trained on, causing prediction errors.
    
    Args:
        model_dir (str): The path to the models directory (e.g., 'models').

    Returns:
        tuple: A tuple containing the loaded model object and a list of feature names.

    Raises:
        FileNotFoundError: If no model or feature files are found.
    """
    # 1. Find the latest model file based on modification time
    model_pattern = os.path.join(model_dir, "best_model_*.pkl")
    model_files = glob.glob(model_pattern)
    if not model_files:
        raise FileNotFoundError("No model files found in the 'models' directory.")
    
    latest_model_path = max(model_files, key=os.path.getmtime)
    print(f"Loading latest model: {latest_model_path}")
    
    # 2. Find the latest features file based on modification time
    features_dir = os.path.join(model_dir, "features")
    feature_pattern = os.path.join(features_dir, "training_features_*.pkl")
    feature_files = glob.glob(feature_pattern)
    if not feature_files:
        raise FileNotFoundError("No feature files found in the 'models/features' directory.")
        
    latest_feature_path = max(feature_files, key=os.path.getmtime)
    print(f"Loading latest features from: {latest_feature_path}")

    # 3. Load the model and features
    try:
        model = joblib.load(latest_model_path)
        training_features = joblib.load(latest_feature_path)
    except Exception as e:
        raise RuntimeError(f"Error loading model or features: {e}")

    return model, training_features