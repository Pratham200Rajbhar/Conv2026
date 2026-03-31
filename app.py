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
import requests
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

OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
SOILGRIDS_PROPERTIES_QUERY_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

@st.cache_data(ttl=60 * 60 * 24)
def geocode_place_candidates(place_name: str, count: int = 5):
    params = {
        "name": place_name,
        "count": max(1, min(int(count), 25)),
        "language": "en",
        "format": "json",
        "countryCode": "IN",
    }
    r = requests.get(OPEN_METEO_GEOCODE_URL, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    return data.get("results") or []

def _best_geocode_candidate(candidates: list[dict], state_hint: str | None):
    if not candidates:
        return None
    if not state_hint:
        return candidates[0]
    state_hint_norm = state_hint.strip().lower()
    for c in candidates:
        admin1 = (c.get("admin1") or "").strip().lower()
        if admin1 and admin1 == state_hint_norm:
            return c
    return candidates[0]

def geocode_place(place_name: str, state_hint: str | None = None):
    candidates = geocode_place_candidates(place_name, count=10)
    top = _best_geocode_candidate(candidates, state_hint=state_hint)
    if not top:
        return None
    return {
        "name": top.get("name"),
        "admin1": top.get("admin1"),
        "country": top.get("country"),
        "latitude": top.get("latitude"),
        "longitude": top.get("longitude"),
        "timezone": top.get("timezone"),
        "raw": top,
        "candidates": candidates,
    }

@st.cache_data(ttl=60 * 15)
def fetch_open_meteo_hourly(latitude: float, longitude: float):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m",
                "shortwave_radiation",
            ]
        ),
        "timezone": "auto",
        "forecast_days": 1,
    }
    r = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def _hourly_value_at_index(payload: dict, var: str, idx: int):
    hourly = payload.get("hourly") or {}
    arr = hourly.get(var) or []
    if 0 <= idx < len(arr):
        return arr[idx]
    return None

@st.cache_data(ttl=60 * 60 * 24)
def fetch_soilgrids_properties(latitude: float, longitude: float):
    # SoilGrids is occasionally rate-limited / down; keep this resilient.
    params = {
        "lat": latitude,
        "lon": longitude,
        "property": ["phh2o", "nitrogen", "soc", "clay", "sand", "silt"],
        "depth": ["0-5cm"],
        "value": ["mean"],
    }
    r = requests.get(SOILGRIDS_PROPERTIES_QUERY_URL, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def _soilgrids_mean_value(payload: dict, prop: str):
    """
    Returns the mean value at 0-5cm if present.
    SoilGrids units vary by property (see docs). For phh2o, values are typically pH*10.
    """
    props = payload.get("properties") or {}
    layers = props.get("layers") or []
    for layer in layers:
        if (layer.get("name") or "").lower() != prop.lower():
            continue
        depths = layer.get("depths") or []
        for d in depths:
            if (d.get("label") or "").lower() != "0-5cm":
                continue
            values = d.get("values") or {}
            if "mean" in values:
                return values["mean"]
    return None

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

        st.divider()
        st.caption("Make location more specific for better auto-fetch.")
        location_mode = st.radio("Location mode", ["District/State", "Precise coordinates (recommended)"], horizontal=False)

        if "latitude" not in st.session_state:
            st.session_state["latitude"] = None
        if "longitude" not in st.session_state:
            st.session_state["longitude"] = None

        if location_mode == "Precise coordinates (recommended)":
            st.session_state["latitude"] = st.number_input(
                "Latitude",
                value=float(st.session_state["latitude"]) if st.session_state["latitude"] is not None else 23.0225,
                format="%.6f",
                help="Tip: paste from Google Maps.",
            )
            st.session_state["longitude"] = st.number_input(
                "Longitude",
                value=float(st.session_state["longitude"]) if st.session_state["longitude"] is not None else 72.5714,
                format="%.6f",
                help="Tip: paste from Google Maps.",
            )
        else:
            place_query = st.text_input(
                "Search place (try: 'District, State')",
                value=f"{district}, {state}, India",
            )
            if st.button("Find coordinates for this place"):
                try:
                    candidates = geocode_place_candidates(place_query, count=10)
                    if not candidates:
                        st.warning("No geocoding results. Try adding 'India' or a nearby city name.")
                    else:
                        preferred = _best_geocode_candidate(candidates, state_hint=state)
                        preferred_id = preferred.get("id") if isinstance(preferred, dict) else None
                        opts = []
                        for c in candidates:
                            label = ", ".join(
                                [p for p in [c.get("name"), c.get("admin1"), c.get("country")] if p]
                            )
                            opts.append((label, c))

                        default_idx = 0
                        if preferred_id is not None:
                            for i, (_, c) in enumerate(opts):
                                if c.get("id") == preferred_id:
                                    default_idx = i
                                    break

                        chosen_label = st.selectbox(
                            "Select the best match",
                            options=[o[0] for o in opts],
                            index=default_idx,
                        )
                        chosen = None
                        for lbl, c in opts:
                            if lbl == chosen_label:
                                chosen = c
                                break
                        if chosen and chosen.get("latitude") is not None and chosen.get("longitude") is not None:
                            st.session_state["latitude"] = float(chosen["latitude"])
                            st.session_state["longitude"] = float(chosen["longitude"])
                            st.success("Coordinates saved. You can now fetch weather/soil reliably.")
                except requests.RequestException:
                    st.warning("Could not reach the geocoding service right now.")
                except Exception:
                    st.warning("Geocoding failed. Try a different query.")

    with col2:
        st.subheader("🧪 Soil & Nutrition")
        st.caption("Tip: Auto-fill soil pH and soil properties from SoilGrids when coordinates are available.")

        if "soil_ph" not in st.session_state:
            st.session_state["soil_ph"] = 7.0

        auto_soil = st.checkbox("Auto-fill soil from live data (SoilGrids)", value=False)
        if auto_soil and st.button("Fetch soil now"):
            lat = st.session_state.get("latitude")
            lon = st.session_state.get("longitude")
            if lat is None or lon is None:
                st.warning("Please set precise coordinates (Latitude/Longitude) first.")
            else:
                try:
                    soil = fetch_soilgrids_properties(float(lat), float(lon))
                    phh2o = _soilgrids_mean_value(soil, "phh2o")
                    if phh2o is not None:
                        # SoilGrids phh2o is commonly pH*10 in many layers
                        ph_value = float(phh2o) / 10.0 if float(phh2o) > 14 else float(phh2o)
                        st.session_state["soil_ph"] = max(0.0, min(14.0, ph_value))
                        st.success("Soil pH loaded.")

                    n_total = _soilgrids_mean_value(soil, "nitrogen")
                    soc = _soilgrids_mean_value(soil, "soc")
                    clay = _soilgrids_mean_value(soil, "clay")
                    sand = _soilgrids_mean_value(soil, "sand")
                    silt = _soilgrids_mean_value(soil, "silt")

                    with st.expander("SoilGrids details (0-5cm)"):
                        st.write(
                            {
                                "nitrogen": n_total,
                                "soc": soc,
                                "clay": clay,
                                "sand": sand,
                                "silt": silt,
                                "note": "Units depend on SoilGrids layer definitions.",
                            }
                        )
                except requests.RequestException:
                    st.warning("Could not reach SoilGrids right now (it can be rate-limited / down).")
                except Exception:
                    st.warning("Soil fetch failed. Using manual inputs.")

        n_req = st.slider("Nitrogen Requirement (kg/ha)", 0, 300, 100)
        p_req = st.slider("Phosphorus Requirement (kg/ha)", 0, 300, 50)
        k_req = st.slider("Potassium Requirement (kg/ha)", 0, 300, 50)
        ph = st.slider("Soil pH", 0.0, 14.0, float(st.session_state["soil_ph"]), key="soil_ph")

    with col3:
        st.subheader("☁️ Climate")
        st.caption("Tip: Auto-fill these from live weather (Open‑Meteo).")

        if "temp_c" not in st.session_state:
            st.session_state["temp_c"] = 25
        if "hum_pct" not in st.session_state:
            st.session_state["hum_pct"] = 60
        if "rain_mm" not in st.session_state:
            st.session_state["rain_mm"] = 500.0
        if "wind_ms" not in st.session_state:
            st.session_state["wind_ms"] = 3.0
        if "solar_mj" not in st.session_state:
            st.session_state["solar_mj"] = 15.0

        auto_weather = st.checkbox("Auto-fill climate from live weather", value=False)
        if auto_weather and st.button("Fetch live weather now"):
            try:
                lat = st.session_state.get("latitude")
                lon = st.session_state.get("longitude")

                if lat is None or lon is None:
                    # Try a few queries because district names can be ambiguous.
                    queries = [
                        f"{district}, {state}, India",
                        f"{district} district, {state}, India",
                        f"{district}, India",
                    ]
                    geo = None
                    for q in queries:
                        geo = geocode_place(q, state_hint=state)
                        if geo and geo.get("latitude") is not None and geo.get("longitude") is not None:
                            break
                    if not geo or geo.get("latitude") is None or geo.get("longitude") is None:
                        st.warning("Could not find coordinates. Add precise Latitude/Longitude for best results.")
                        geo = None
                    else:
                        lat = float(geo["latitude"])
                        lon = float(geo["longitude"])
                        st.session_state["latitude"] = lat
                        st.session_state["longitude"] = lon

                if lat is None or lon is None:
                    st.warning("Using manual climate inputs.")
                else:
                    wx = fetch_open_meteo_hourly(float(lat), float(lon))
                    times = (wx.get("hourly") or {}).get("time") or []
                    idx = 0

                    st.session_state["temp_c"] = float(_hourly_value_at_index(wx, "temperature_2m", idx) or st.session_state["temp_c"])
                    st.session_state["hum_pct"] = int(round(float(_hourly_value_at_index(wx, "relative_humidity_2m", idx) or st.session_state["hum_pct"])))
                    st.session_state["rain_mm"] = float(_hourly_value_at_index(wx, "precipitation", idx) or st.session_state["rain_mm"])
                    st.session_state["wind_ms"] = float(_hourly_value_at_index(wx, "wind_speed_10m", idx) or st.session_state["wind_ms"])

                    # NOTE: Open‑Meteo shortwave_radiation is hourly W/m². We keep it as a rough proxy
                    # by mapping the current hour value into the existing "solar" field.
                    st.session_state["solar_mj"] = float(_hourly_value_at_index(wx, "shortwave_radiation", idx) or st.session_state["solar_mj"])

                    if st.session_state.get("latitude") is not None and st.session_state.get("longitude") is not None:
                        st.success("Weather loaded for saved coordinates.")
                    if times:
                        st.caption(f"Using forecast hour: {times[idx]}")
            except requests.RequestException:
                st.warning("Could not reach Open‑Meteo. Using manual climate inputs.")
            except Exception:
                st.warning("Weather fetch failed. Using manual climate inputs.")

        temp = st.slider("Temperature (°C)", -10, 60, int(round(float(st.session_state["temp_c"]))), key="temp_c")
        hum = st.slider("Humidity (%)", 0, 100, int(st.session_state["hum_pct"]), key="hum_pct")
        rain = st.number_input("Rainfall (mm)", min_value=0.0, value=float(st.session_state["rain_mm"]), key="rain_mm")
        wind = st.slider("Wind Speed (m/s)", 0.0, 50.0, float(st.session_state["wind_ms"]), key="wind_ms")
        solar = st.slider("Solar Radiation (MJ/m2/day)", 0.0, 40.0, float(st.session_state["solar_mj"]), key="solar_mj")

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
