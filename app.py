import os
# Suppress TensorFlow logging and oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import cv2
import xgboost as xgb
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from PIL import Image
from tensorflow.keras.models import load_model

# Suppress common warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Page Configuration
st.set_page_config(
    page_title="Agri-Smart: Yield & Disease AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #4CAF50;
    }
    .disease-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

# Paths
YIELD_MODELS_DIR = "/work/Convergance Hackathon/models/ML_Yield_Project/models"
DISEASE_MODELS_DIR = "/work/Convergance Hackathon/models/Train_model"
CONFIG_PATH = "/work/Convergance Hackathon/config.json"

# Load Configuration
@st.cache_resource
def load_app_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

config = load_app_config()

# Load Yield Models
@st.cache_resource
def load_yield_resources():
    def load_pkl(filename):
        path = os.path.join(YIELD_MODELS_DIR, filename)
        if not os.path.exists(path):
            path = os.path.join("/work/Convergance Hackathon/models", filename)
        if not os.path.exists(path):
            return None
        with open(path, 'rb') as f:
            return pickle.load(f)

    return {
        "rf": load_pkl("rf_model.pkl"),
        "xgb": load_pkl("xgb_model.pkl"),
        "le_state": load_pkl("le_state.pkl"),
        "le_dist": load_pkl("le_dist.pkl"),
        "le_crop": load_pkl("le_crop.pkl"),
        "scaler": load_pkl("scaler.pkl")
    }

yield_res = load_yield_resources()

# Load Disease Models Label Map
DISEASE_LABELS = {
    "Brinjal": ["Bacterial Wilt", "Cercospora Leaf Spot", "Healthy", "Mosaic", "Phomopsis Leaf Blight"],
    "Castor": ["Alternaria Leaf Blight", "Bacterial Leaf Blight", "Cercospora Leaf Spot", "Healthy", "Leaf Curv Virus"],
    "Cumin": ["Alternaria Blight", "Healthy", "Wilt"],
    "Guava": ["Anthracnose", "Bacterial Blight", "Healthy", "Red Rust", "Wilt"],
    "Papaya": ["Healthy", "Leaf Spot", "Powdery Mildew", "Ring Spot Virus"]
}

@st.cache_resource
def load_disease_model(crop):
    path = os.path.join(DISEASE_MODELS_DIR, f"{crop}_model.keras")
    return load_model(path)

# UI Layout
st.sidebar.title("🌿 Agri-Smart AI")
st.sidebar.markdown("Helping farmers with data-driven decisions.")
mode = st.sidebar.radio("Select Application", ["🌾 Crop Yield Prediction", "🔬 Plant Disease Classification"])

if mode == "🌾 Crop Yield Prediction":
    st.header("🌾 Crop Yield Prediction Dashboard")
    st.markdown("Enter details below to predict the estimated crop yield.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📍 Geography")
        state = st.selectbox("State Name", config["states"])
        district = st.selectbox("District Name", config["districts"])
        crop = st.selectbox("Crop", config["crops"])
        year = st.number_input("Year", min_value=2000, max_value=2030, value=2024)
        area = st.number_input("Area (ha)", min_value=0.1, value=1.0)

    with col2:
        st.subheader("🧪 Soil & Nutrition")
        n_req = st.slider("Nitrogen Requirement (kg/ha)", 0, 300, 100)
        p_req = st.slider("Phosphorus Requirement (kg/ha)", 0, 300, 50)
        k_req = st.slider("Potassium Requirement (kg/ha)", 0, 300, 50)
        ph = st.slider("Soil pH", 0.0, 14.0, 7.0)

    with col3:
        st.subheader("☁️ Climate")
        temp = st.slider("Temperature (°C)", -10, 60, 25)
        hum = st.slider("Humidity (%)", 0, 100, 60)
        rain = st.number_input("Rainfall (mm)", min_value=0.0, value=500.0)
        wind = st.slider("Wind Speed (m/s)", 0.0, 50.0, 3.0)
        solar = st.slider("Solar Radiation (MJ/m2/day)", 0.0, 40.0, 15.0)

    if st.button("🚀 Predict Yield"):
        try:
            # Encode Categoricals
            state_idx = yield_res["le_state"].transform([state])[0]
            dist_idx = yield_res["le_dist"].transform([district])[0]
            crop_idx = yield_res["le_crop"].transform([crop])[0]
            
            # Prepare Input Data for Scaling (14 features)
            input_data = [
                year, state_idx, dist_idx, crop_idx, area,
                n_req, p_req, k_req, temp, hum, ph, rain, wind, solar
            ]
            
            # Feature names for Scaler
            feature_names_14 = [
                'Year', 'State Name', 'Dist Name', 'Crop', 'Area_ha', 
                'N_req_kg_per_ha', 'P_req_kg_per_ha', 'K_req_kg_per_ha', 
                'Temperature_C', 'Humidity_%', 'pH', 'Rainfall_mm', 
                'Wind_Speed_m_s', 'Solar_Radiation_MJ_m2_day'
            ]
            
            # Feature names for Models (11 features)
            feature_names_11 = [
                'Year', 'State Name', 'Dist Name', 'Crop', 'Area_ha', 
                'Temperature_C', 'Humidity_%', 'pH', 'Rainfall_mm', 
                'Wind_Speed_m_s', 'Solar_Radiation_MJ_m2_day'
            ]

            # Scale using DataFrame to avoid Feature Name warnings
            input_df_14 = pd.DataFrame([input_data], columns=feature_names_14)
            input_scaled_df = pd.DataFrame(yield_res["scaler"].transform(input_df_14), columns=feature_names_14)
            
            # Drop nutrient requirements (indices 5, 6, 7) to match the model's 11 features
            input_model_df = input_scaled_df[feature_names_11]
            
            # Predict
            rf_pred = yield_res["rf"].predict(input_model_df)[0]
            
            # XGBoost Predict
            dmat = xgb.DMatrix(input_model_df)
            xgb_pred = yield_res["xgb"].predict(dmat)[0]
            
            st.divider()
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.markdown(f'<div class="prediction-card"><h3>Random Forest Prediction</h3><h2>{rf_pred:.2f} kg/ha</h2></div>', unsafe_allow_html=True)
            
            with res_col2:
                st.markdown(f'<div class="prediction-card"><h3>XGBoost Prediction</h3><h2>{xgb_pred:.2f} kg/ha</h2></div>', unsafe_allow_html=True)
                
            st.info(f"Average Predicted Yield: {(rf_pred + xgb_pred)/2:.2f} kg/ha")
            
        except Exception as e:
            st.error(f"Error in prediction: {e}")

else:
    st.header("🔬 Plant Disease Classification")
    st.markdown("Upload a leaf image to diagnose potential diseases.")
    
    crop_type = st.selectbox("Select Crop Type", list(DISEASE_LABELS.keys()))
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Leaf Image', width=300)
        
        if st.button("🔍 Diagnose Disease"):
            with st.spinner("Processing image..."):
                try:
                    # Load model
                    model = load_disease_model(crop_type)
                    
                    # Preprocess
                    img_array = np.array(image.convert('RGB'))
                    img_resized = cv2.resize(img_array, (224, 224))
                    img_normalized = img_resized / 255.0
                    img_batch = np.expand_dims(img_normalized, axis=0)
                    
                    # Predict
                    preds = model.predict(img_batch)
                    class_idx = np.argmax(preds[0])
                    confidence = preds[0][class_idx]
                    
                    label = DISEASE_LABELS[crop_type][class_idx]
                    
                    st.divider()
                    st.markdown(f'<div class="disease-card"><h3>Diagnosis: {label}</h3><h2>Confidence: {confidence*100:.2f}%</h2></div>', unsafe_allow_html=True)
                    
                    if "Healthy" in label:
                        st.success("The leaf appears to be healthy! Keep up the good work.")
                    else:
                        st.warning(f"Detected disease: {label}. Please consult an agronomist for treatment.")
                        
                except Exception as e:
                    st.error(f"Error in classification: {e}")

st.sidebar.divider()
st.sidebar.caption("© 2026 Agri-Smart AI. Developed for HT202602 Hackathon.")
