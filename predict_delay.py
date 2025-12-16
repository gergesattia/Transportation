import os
import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

def load_model(model_path=None):
    """Load the trained model.

    If `model_path` is not provided, try common filenames in the script directory.
    Raises FileNotFoundError if no suitable model file is found.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    candidates = []
    if model_path:
        candidates.append(model_path)

    # common candidate locations (script dir and parent)
    candidates += [
        os.path.join(script_dir, 'best_delay_model.pkl'),
        os.path.join(script_dir, 'trained_model.pkl'),
        os.path.join(script_dir, 'model.pkl'),
        os.path.join(script_dir, 'best_model.pkl'),
        os.path.join(script_dir, '..', 'best_delay_model.pkl'),
    ]

    for path in candidates:
        if path and os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                return model
            except Exception as e:
                raise RuntimeError(f"Failed to load model from {path}: {e}") from e

    raise FileNotFoundError(f"No model file found. Tried: {', '.join(candidates)}")

def predict_delay(model, features_dict):
    """
    Predict trip delay
    
    Parameters:
    -----------
    model: trained model
    features_dict: dictionary containing the required features
    
    Returns:
    --------
    predicted_delay: expected delay in minutes
    """
    # Convert dictionary to DataFrame
    df = pd.DataFrame([features_dict])
    
    # Prediction
    prediction = model.predict(df)[0]
    
    return max(0, prediction)  # Delay cannot be negative

def main():
    """Example usage"""
    print("🚌 Trip Delay Prediction Model")
    print("=" * 50)
    
    # Load the model with error handling
    try:
        model = load_model()
        print("✓ Model loaded successfully")
    except FileNotFoundError as e:
        print("✖ Model file not found:", e)
        print("Hint: place the model file (e.g. 'best_delay_model.pkl') in the same folder as this script.")
        sys.exit(1)
    except Exception as e:
        print("✖ Failed to load model:", e)
        sys.exit(1)
    
    # Example of required features
    example_features = {
        'hour': 18,
        'day_of_week': 2,  # Wednesday
        'month': 1,
        'is_weekend': 0,
        'day_period_encoded': 2,  # Evening
        'time_category_en_encoded': 2,  # Evening
        'passenger_count_final': 50,
        'passenger_load_index': 1.5,
        'prev_delay': 15,
        'rush_period': 1,
        'route_frequency': 69,
        'speed_proxy': 0.5,
        'weather_severity': 0.0,
        'distance_change': 0.5,
        'latitude_clean': 25.5,
        'longitude_clean': 32.0,
        'weather_en_encoded': 0,  # Clear
        'passenger_level_encoded': 1,  # Low
    }
    
    # Predict
    predicted_delay = predict_delay(model, example_features)
    
    print(f"\n📊 Results:")
    print(f"  Expected delay: {predicted_delay:.2f} minutes")
    print(f"  Expected delay: {predicted_delay / 60:.2f} hours")
    
    # Categorize delay severity
    if predicted_delay < 5:
        severity = "Very minor delay"
    elif predicted_delay < 15:
        severity = "Minor delay"
    elif predicted_delay < 30:
        severity = "Moderate delay"
    elif predicted_delay < 60:
        severity = "Major delay"
    else:
        severity = "Very severe delay"
    
    print(f"  Severity level: {severity}")

if __name__ == "__main__":
    main()
