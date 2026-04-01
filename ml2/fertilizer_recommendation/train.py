import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Paths
DATASET_PATH = r"d:\Chirag\Hackathon\Conv2026\ml2\fertilizer_recommendation\dataset\Fertilizer Prediction (1).csv"
OUTPUT_DIR = r"d:\Chirag\Hackathon\Conv2026\ml2\fertilizer_recommendation"

def train_model():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    
    # Feature engineering / Preprocessing
    # The columns are: Temparature, Humidity, Moisture, Soil Type, Crop Type, Nitrogen, Potassium, Phosphorous, Fertilizer Name
    
    # Check for missing values
    if df.isnull().values.any():
        print("Missing values found, dropping them.")
        df = df.dropna()
    
    # Label Encoding for categorical features
    le_soil = LabelEncoder()
    df['Soil Type'] = le_soil.fit_transform(df['Soil Type'])
    
    le_crop = LabelEncoder()
    df['Crop Type'] = le_crop.fit_transform(df['Crop Type'])
    
    le_fertilizer = LabelEncoder()
    df['Fertilizer Name'] = le_fertilizer.fit_transform(df['Fertilizer Name'])
    
    # X and y
    X = df.drop('Fertilizer Name', axis=1)
    y = df['Fertilizer Name']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    print("Training Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy Score: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le_fertilizer.classes_))
    
    # Save model and encoders
    print(f"Saving artifacts to {OUTPUT_DIR}...")
    joblib.dump(rf, os.path.join(OUTPUT_DIR, 'fertilizer_rf_model.pkl'))
    joblib.dump(le_soil, os.path.join(OUTPUT_DIR, 'soil_encoder.pkl'))
    joblib.dump(le_crop, os.path.join(OUTPUT_DIR, 'crop_encoder.pkl'))
    joblib.dump(le_fertilizer, os.path.join(OUTPUT_DIR, 'fertilizer_encoder.pkl'))
    
    return accuracy

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"Error during training: {e}")
