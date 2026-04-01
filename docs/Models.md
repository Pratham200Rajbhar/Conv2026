<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# give me complete table of features and models and datasets use in project

Based on all your uploaded project files, here is the complete reference table for every feature, model, and dataset across all modules of your AgriAI project:

***

## 🌾 Crop Yield Prediction

| Field | Details |
| :-- | :-- |
| **Input Features** | `N`, `P`, `K`, `pH`, `Temperature`, `Humidity`, `Rainfall`, `Solar_Radiation`, `State`, `District`, `Season`, `Crop_Type` [^1] |
| **Target Variable** | `Yield_kg_per_ha` [^1] |
| **Models Used** | Random Forest Regressor (R²=0.866), XGBoost Regressor (R²=0.765) [^1] |
| **Best Model** | ✅ Random Forest (`rf_model.pkl`) [^1] |
| **Preprocessing** | `LabelEncoder` for categorical, `StandardScaler` for numerical, missing value imputation [^1] |
| **Saved Artifacts** | `rf_model.pkl`, `xgb_model.pkl`, `scaler.pkl`, `label_encoders.pkl` [^1] |
| **Notebooks** | `01_Data_Preprocessing` → `02_Feature_Engineering` → `03_RF_Training` → `04_XGB_Training` → `05_Comparison` → `06_Inference` [^1] |
| **Dataset** | ICRISAT District-Level DB + IMD Weather (State/District yield + temperature, rainfall, humidity 1990–2015) [^1] |


***

## 🍃 Plant Disease Classification

| Crop | Diseases Covered | Test Accuracy | Model |
| :-- | :-- | :-- | :-- |
| **Cumin** | Blight, Powdery Mildew, Healthy | 99% [^1] | MobileNetV2 |
| **Brinjal** | Early Blight, Leaf Spot, Healthy | 94% [^1] | MobileNetV2 |
| **Guava** | Anthracnose, Canker, Healthy | 88% [^1] | MobileNetV2 |
| **Castor** | Leaf Spot, Blight, Healthy | 80% [^1] | MobileNetV2 |
| **Papaya** | Ring Virus, Leaf Curl, Healthy | 72% [^1] | MobileNetV2 |

| Field | Details |
| :-- | :-- |
| **Input Features** | 224×224 RGB leaf images [^1] |
| **Architecture** | MobileNetV2 (ImageNet weights, `include_top=False`) + GlobalAveragePooling2D + Dense(128, ReLU) + Dropout(0.3) + Softmax [^1] |
| **Training Config** | Adam optimizer, `categorical_crossentropy` loss, 20 epochs, batch size 32, EarlyStopping + ReduceLROnPlateau [^1] |
| **Saved Artifacts** | `cumin_model.keras`, `brinjal_model.keras`, `guava_model.keras`, `castor_model.keras`, `papaya_model.keras` [^1] |
| **Dataset** | Custom curated dataset — 5 crop × 3 class folders; PlantVillage + field images [^1] |


***

## 🌱 Crop Recommendation Engine

| Field | Details |
| :-- | :-- |
| **Input Features** | `N`, `P`, `K`, `pH`, `Temperature`, `Humidity`, `Rainfall` [^2] |
| **Target Variable** | Crop Name (Rice, Wheat, Cotton, Maize, etc.) |
| **Recommended Model** | Random Forest Classifier or Naive Bayes |
| **Expected Accuracy** | ~97–99% |
| **Dataset** | [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) — 2,200 rows, 22 crop classes |


***

## 🧪 Fertilizer Recommendation

| Field | Details |
| :-- | :-- |
| **Input Features** | `Soil_Type`, `Crop_Type`, `N_current`, `P_current`, `K_current`, `Temperature`, `Humidity`, `Moisture` [^2] |
| **Target Variable** | Fertilizer Name (Urea, DAP, 14-35-14, etc.) |
| **Recommended Model** | Decision Tree Classifier or Random Forest Classifier |
| **Expected Accuracy** | ~92–95% |
| **Dataset** | [Kaggle Fertilizer Prediction Dataset](https://www.kaggle.com/datasets/gdabhishek/fertilizer-prediction) — 10 fertilizer classes |


***

## 💧 Irrigation Scheduling

| Field | Details |
| :-- | :-- |
| **Input Features** | `Temperature`, `Humidity`, `Wind_Speed`, `Solar_Radiation`, `Soil_Moisture`, `Rainfall`, `Crop_Stage` [^2] |
| **Target Variable** | `Irrigation_mm_per_day` or binary irrigate/skip |
| **Recommended Model** | XGBoost Regressor (reuse existing pipeline) OR rule-based threshold (soil moisture < 30% → irrigate) |
| **Dataset** | FAO AquaCrop data or Penman-Monteith ET₀ formula derived from existing weather columns [^1] |


***

## 🪲 Pest / Disease Risk Alert

| Field | Details |
| :-- | :-- |
| **Input Features** | `Temperature`, `Humidity`, `Rainfall`, `Wind_Speed`, `Season`, `Crop_Type` [^2] |
| **Target Variable** | `pest_risk` — Binary (0 = Low, 1 = High) |
| **Recommended Model** | Logistic Regression or Gradient Boosting Classifier |
| **Expected Accuracy** | ~85–90% |
| **Dataset** | Derive from existing `cleaned_dataset.csv` weather columns [^1] + ICAR pest outbreak records |


***

## 📈 Market Price Prediction

| Field | Details |
| :-- | :-- |
| **Input Features** | `Date`, `Crop`, `District/State`, `Historical_Price` [^2][^3] |
| **Target Variable** | `Price_per_quintal` (7/14/30-day forecast) |
| **Recommended Model** | Facebook Prophet (fastest) or LSTM (deep learning) |
| **Dataset** | Agmarknet API (live mandi prices) or [data.gov.in commodity prices](https://data.gov.in) |


***

## 📦 All Datasets at a Glance

| Dataset | Module | Source | Size |
| :-- | :-- | :-- | :-- |
| ICRISAT + IMD Weather Data | Yield Prediction | [bigdata.cgiar.org](https://bigdata.cgiar.org) | 560 districts, 1990–2015 [^1] |
| Custom Leaf Image Dataset | Disease Classification | PlantVillage + field photos | 5 crops × 3 classes [^1] |
| Crop Recommendation Dataset | Crop Recommendation | [Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) | 2,200 rows |
| Fertilizer Prediction Dataset | Fertilizer Rec. | [Kaggle](https://www.kaggle.com/datasets/gdabhishek/fertilizer-prediction) | ~1,000 rows |
| FAO AquaCrop / ET₀ Formula | Irrigation | [fao.org/aquacrop](https://www.fao.org/aquacrop) | Formula-based |
| ICAR Pest Outbreak Records | Pest Risk Alert | [krishi.icar.gov.in](https://krishi.icar.gov.in) | State-level records |
| Agmarknet Mandi Prices | Market Price | [agmarknet.gov.in](https://agmarknet.gov.in) / data.gov.in | Daily, pan-India |


***

## 🔌 APIs Mapped to Each Module

| Module | API | Cost |
| :-- | :-- | :-- |
| Yield Prediction + Irrigation | Open-Meteo (weather auto-fill) | Free [^4] |
| Yield + Pest Risk | SoilGrids REST API (NPK by GPS) | Free [^4] |
| Pest + Irrigation | Google Earth Engine (NDVI, soil moisture) | Free (academic) [^4] |
| Market Price | Agmarknet API (mandi prices) | Free (govt.) [^4] |
| All Modules | OpenWeatherMap (real-time weather) | 1000 req/day free [^4] |

<div align="center">⁂</div>

[^1]: ml1.md

[^2]: List-of-Features.md

[^3]: PRD.md

[^4]: Tech-Stack.md

