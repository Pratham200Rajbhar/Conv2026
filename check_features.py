import pickle
import os

models_dir = "/work/Convergance Hackathon/models/ML_Yield_Project/models"

def load_pkl(filename):
    path = os.path.join(models_dir, filename)
    with open(path, 'rb') as f:
        return pickle.load(f)

rf = load_pkl("rf_model.pkl")
xgb = load_pkl("xgb_model.pkl")
scaler = load_pkl("scaler.pkl")

print(f"Scaler features: {scaler.n_features_in_}")
try:
    print(f"RF features: {rf.n_features_in_}")
except:
    print("RF n_features_in_ not found")

try:
    print(f"XGB features: {xgb.n_features_in_}")
except:
    # For XGBoost, it might be in .feature_names or similar
    try:
        print(f"XGB features: {len(xgb.feature_names_in_)}")
    except:
        print("XGB features not found")
