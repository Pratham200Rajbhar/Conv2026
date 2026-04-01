import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def load_pkl(filename):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        print(f"Error: {filename} not found in {MODELS_DIR}")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

print("Loading yield prediction resources...")
resources = {
    "rf": load_pkl("rf_model.pkl"),
    "xgb": load_pkl("xgb_model.pkl"),
    "le_state": load_pkl("le_state.pkl"),
    "le_dist": load_pkl("le_dist.pkl"),
    "le_crop": load_pkl("le_crop.pkl"),
    "scaler": load_pkl("scaler.pkl")
}

if all(resources.values()):
    print("All models loaded successfully!")
    
    # Dummy verification
    try:
        # Sample features (values are just for testing)
        feature_names = [
            'Year', 'State Name', 'Dist Name', 'Crop', 'Area_ha',
            'Temperature_C', 'Humidity_%', 'pH', 'Rainfall_mm',
            'Wind_Speed_m_s', 'Solar_Radiation_MJ_m2_day'
        ]
        
        # Create a dummy input with valid categories from label encoders
        state = resources["le_state"].classes_[0]
        dist = resources["le_dist"].classes_[0]
        crop = resources["le_crop"].classes_[0]
        
        state_idx = resources["le_state"].transform([state])[0]
        dist_idx = resources["le_dist"].transform([dist])[0]
        crop_idx = resources["le_crop"].transform([crop])[0]
        
        dummy_input = [2024, state_idx, dist_idx, crop_idx, 1.0, 25.0, 60.0, 7.0, 500.0, 3.0, 15.0]
        
        input_df = pd.DataFrame([dummy_input], columns=feature_names)
        input_scaled = resources["scaler"].transform(input_df)
        input_model_df = pd.DataFrame(input_scaled, columns=feature_names)
        
        # RF Prediction
        rf_pred = resources["rf"].predict(input_model_df)[0]
        print(f"Random Forest Prediction: {rf_pred:.2f} kg/ha")
        
        # XGB Prediction
        dmat = xgb.DMatrix(input_model_df)
        xgb_pred = resources["xgb"].predict(dmat)[0]
        print(f"XGBoost Prediction: {xgb_pred:.2f} kg/ha")
        
        print("\nVerification Successful!")
    except Exception as e:
        print(f"\nVerification failed during prediction: {e}")
else:
    print("Failed to load one or more models.")
