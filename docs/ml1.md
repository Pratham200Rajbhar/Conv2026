# ML1 Complete Technical Documentation

## 1. Overview

ML1 is a dual-track machine learning workspace containing:

1. A plant disease classification system based on image data and MobileNetV2 transfer learning.
2. A crop yield prediction system based on tabular agro-climatic and soil features using Random Forest and XGBoost.

This documentation describes the current implementation state as found in the workspace, including datasets, notebooks, generated artifacts, observed behavior, and known gaps.

## 2. Workspace Scope

Primary location: `ML1/`

High-level structure:

- `ML1/Datasets/`: Source image dataset by crop and disease state.
- `ML1/split_dataset/`: Train/validation/test image split used for CNN training and evaluation.
- `ML1/notebooks/`: Image-classification notebook pipeline.
- `ML1/Train_model/`: Trained Keras models (per crop).
- `ML1/ML_Yield_Project/`: Tabular yield prediction subproject.
- `ML1/models/`: Additional model artifact copy.
- `ML1/result_/`: Empty placeholder directory.

## 3. Problem Domain Coverage

### 3.1 Image Classification Domain

Crops covered:

- Brinjal
- Castor
- Cumin
- Guava
- Papaya

For each crop, labels include:

- `Healthy`
- Crop-specific disease labels under `Unhealthy/`

### 3.2 Yield Prediction Domain

Target variable:

- `Yield_kg_per_ha`

Input space includes:

- Geographic columns (state, district)
- Crop category
- Area
- Weather (temperature, humidity, rainfall, wind, solar radiation)
- Soil/chemical variables (`pH`, and nutrient requirements columns)

## 4. Dataset Documentation

## 4.1 Source Image Dataset (`ML1/Datasets`)

Directory design:

- `Datasets/<Crop>/Healthy/*`
- `Datasets/<Crop>/Unhealthy/<Disease>/*`

Class map by crop:

- Brinjal: `Healthy`, `Bacrerial_Wilt`, `Cercospora_Leaf_spot`, `Mosaic`, `Phomopsis_Leaf_Blight`
- Castor: `Healthy`, `Alternaria_Leaf_Blight`, `Bacterial_Leaf_Blight`, `Cercospora_Leaf_Spot`, `leaf_curv_virus`
- Cumin: `Healthy`, `Alternaria_Blight`, `Wilt`
- Guava: `Healthy`, `Anthracnose`, `Bacterial_Blight`, `Red_Rust`, `Wilt`
- Papaya: `Healthy`, `Leaf_Spot`, `Powdery_Mildew`, `Ring_Spot_Virus`

Source image counts (current snapshot):

- Brinjal: 152
- Castor: 112
- Cumin: 52
- Guava: 134
- Papaya: 77
- Total source images: 527

## 4.2 Split Dataset (`ML1/split_dataset`)

Structure:

- `split_dataset/train/<Crop>/<Class>/*`
- `split_dataset/val/<Crop>/<Class>/*`
- `split_dataset/test/<Crop>/<Class>/*`

Global split counts:

- Train: 520
- Val: 251
- Test: 277
- Total: 1048

Per-crop totals:

- Brinjal: 310
- Castor: 222
- Cumin: 106
- Guava: 259
- Papaya: 151

Detailed class distribution:

### Brinjal
- Bacrerial_Wilt: train 16, val 9, test 9, total 34
- Cercospora_Leaf_spot: train 47, val 24, test 26, total 97
- Healthy: train 54, val 26, test 30, total 110
- Mosaic: train 14, val 7, test 10, total 31
- Phomopsis_Leaf_Blight: train 20, val 8, test 10, total 38

### Castor
- Alternaria_Leaf_Blight: train 20, val 9, test 14, total 43
- Bacterial_Leaf_Blight: train 20, val 10, test 10, total 40
- Cercospora_Leaf_Spot: train 22, val 9, test 9, total 40
- Healthy: train 35, val 17, test 17, total 69
- leaf_curv_virus: train 14, val 7, test 9, total 30

### Cumin
- Alternaria_Blight: train 17, val 9, test 10, total 36
- Healthy: train 24, val 9, test 14, total 47
- Wilt: train 11, val 6, test 6, total 23

### Guava
- Anthracnose: train 32, val 17, test 15, total 64
- Bacterial_Blight: train 16, val 7, test 7, total 30
- Healthy: train 45, val 23, test 21, total 89
- Red_Rust: train 14, val 7, test 7, total 28
- Wilt: train 24, val 13, test 11, total 48

### Papaya
- Healthy: train 33, val 17, test 19, total 69
- Leaf_Spot: train 13, val 7, test 7, total 27
- Powdery_Mildew: train 14, val 4, test 7, total 25
- Ring_Spot_Virus: train 15, val 6, test 9, total 30

Important data lineage note:

- Source dataset total (527) and split total (1048) are not aligned, implying that the split dataset likely includes additional historical/manual/augmented images or was created from a different source snapshot.

## 5. Image Classification Pipeline Documentation

Notebook path set:

- `ML1/notebooks/01_Data_Preparation.ipynb`
- `ML1/notebooks/02_MobileNetV2_Training.ipynb`
- `ML1/notebooks/03_model_evalution.ipynb`
- `ML1/notebooks/04_web_integration_test.ipynb`

## 5.1 Data Preparation (`01_Data_Preparation.ipynb`)

Functional behavior:

- Mounts Google Drive (Colab-centric execution).
- Iterates crop directory tree and copies files into `train/val/test` with ratio 70/15/15.
- Handles healthy classes and unhealthy disease classes separately.
- Uses `random.shuffle` + file copy operations.

Implementation implications:

- No fixed random seed in split logic -> split is non-deterministic across runs.
- Existing typo class naming (`Bacrerial_Wilt`) propagates into model labels and output classes.

## 5.2 MobileNetV2 Training (`02_MobileNetV2_Training.ipynb`)

Core strategy:

- Transfer learning with `MobileNetV2` (`weights='imagenet'`, `include_top=False`).
- Input image size: 224x224.
- Batch size: 16.

Data generators:

- Train augmentation includes:
  - Rescaling
  - Rotation (`rotation_range=20`)
  - Zoom (`zoom_range=0.2`)
  - Horizontal flip
  - Brightness adjustment (`brightness_range=[0.8,1.2]`)

Class imbalance handling:

- Computes class weights using sklearn.
- Includes additional manual downscaling where very high weights (`>3`) are multiplied by 0.7.

Fine-tuning policy:

- Freezes all but last 15 layers of base model.

Classifier head:

- `GlobalAveragePooling2D`
- `Dense(128, relu)`
- `Dropout(0.5)`
- `Dense(num_classes, softmax)`

Optimization and stopping:

- `Adam(learning_rate=1e-4)`
- Loss: categorical cross entropy
- Metric: accuracy
- Early stopping on validation loss (patience 7, restore best weights)
- Epochs: 20

Artifact output:

- Saves crop-specific model into `ML1/Train_model/<Crop>_model.keras`.

Operational pattern:

- `CROP_NAME` is manually changed and notebook rerun per crop.

## 5.3 Model Evaluation (`03_model_evalution.ipynb`)

Behavior:

- Loads a selected crop model.
- Builds test generator and computes predictions.
- Produces:
  - Classification report
  - Confusion matrix
  - Final test accuracy

Embedded summary table in notebook:

- Brinjal: 0.94
- Castor: 0.80
- Cumin: 0.99
- Guava: 0.88
- Papaya: 0.72

Note:

- Test data loading logic appears duplicated in the notebook.
- Static summary table should be treated as snapshot values, not guaranteed live values.

## 5.4 Web Integration Test (`04_web_integration_test.ipynb`)

Purpose:

- Prototype inference flow to support web/app integration testing.

Behavior:

- Loads all 5 crop-specific Keras models into a dictionary.
- Maintains crop-to-class-label mapping dictionary.
- Preprocesses image with OpenCV:
  - Read
  - Resize to 224x224
  - Normalize by 255
  - Add batch dimension
- Predicts class and confidence.
- Supports uploaded image input via Colab upload utility.
- User manually selects crop model for inference.

## 6. Image Model Artifacts

Directory: `ML1/Train_model/`

Artifacts:

- `Brinjal_model.keras`
- `Castor_model.keras`
- `Cumin_model.keras`
- `Guava_model.keras`
- `Papaya_model.keras`

Observed size profile:

- Each model is approximately 19 MB.

Interpretation:

- Similar architecture and weight footprint across crops, indicating per-crop training with common MobileNetV2 backbone and fixed head design.

## 7. Yield Prediction Pipeline Documentation

Notebook path set:

- `ML1/ML_Yield_Project/notebooks/01_Data_Preprocessing.ipynb`
- `ML1/ML_Yield_Project/notebooks/02_Feature_Engineering.ipynb`
- `ML1/ML_Yield_Project/notebooks/03_Model_Training_RF.ipynb`
- `ML1/ML_Yield_Project/notebooks/04_Model_Training_XGBoost.ipynb`
- `ML1/ML_Yield_Project/notebooks/05_Model_Comparison.ipynb`
- `ML1/ML_Yield_Project/notebooks/06_Model_Testing.ipynb`

## 7.1 Data Preprocessing (`01_Data_Preprocessing.ipynb`)

Behavior:

- Downloads Kaggle dataset via `kagglehub`.
- Reads source CSV and performs EDA checks (`shape`, `info`, `describe`, nulls).
- Removes duplicates.
- Drops specific columns:
  - `Dist Code`, `State Code`
  - `Total_N_kg`, `Total_P_kg`, `Total_K_kg`
- Writes cleaned dataset.

Observed implementation note:

- Notebook has duplicated download/setup-style cells.

## 7.2 Feature Engineering (`02_Feature_Engineering.ipynb`)

Behavior:

- Encodes categorical features (`Crop`, `State Name`, `Dist Name`) via `LabelEncoder`.
- Scales features via `StandardScaler`.
- Creates train/test split (80/20, random_state 42).
- Serializes encoders and scaler to `ML1/ML_Yield_Project/models/`.
- Writes split arrays with `np.savez`.
- Includes correlation heatmap analysis.

Important artifact naming issue:

- Arrays saved into hidden filename `.npz` under `ML1/ML_Yield_Project/data/processed/` due path string usage.

## 7.3 Random Forest Training (`03_Model_Training_RF.ipynb`)

Behavior:

- Loads cleaned data.
- Sets target: `Yield_kg_per_ha`.
- Excludes from features:
  - `Yield_kg_per_ha`
  - `N_req_kg_per_ha`
  - `P_req_kg_per_ha`
  - `K_req_kg_per_ha`
- Encodes categorical fields.
- Splits train/test (80/20, random_state 42).
- Trains `RandomForestRegressor` with:
  - `n_estimators=200`
  - `max_depth=15`
  - `random_state=42`
  - `n_jobs=-1`
- Reports RMSE, MAE, and R2.
- Produces feature importance ranking.
- Saves model to `ML1/ML_Yield_Project/models/rf_model.pkl`.

Notebook-reported metrics:

- RMSE: 362.22
- MAE: 215.20
- R2: 0.8581

## 7.4 XGBoost Training (`04_Model_Training_XGBoost.ipynb`)

Behavior:

- Installs `xgboost` in notebook runtime.
- Repeats core preprocessing and train/test split.
- Trains with `XGBRegressor` hyperparameters roughly:
  - `n_estimators=600`
  - `learning_rate=0.03`
  - `max_depth=4`
  - `subsample=0.9`
  - `colsample_bytree=0.9`
  - `gamma=0.1`
  - `reg_alpha=0.5`
  - `reg_lambda=1`
  - `random_state=42`
- Adds a later fix section using `xgb.train` + `DMatrix` + early stopping.
- Saves model as `ML1/ML_Yield_Project/models/xgb_model.pkl`.

Consistency caution:

- After moving to a different training API (`xgb.train`), metric cells may not consistently recompute predictions from the new model object if execution order is mixed.

## 7.5 Model Comparison (`05_Model_Comparison.ipynb`)

Behavior:

- Produces hardcoded comparison table and writes:
  - `ML1/ML_Yield_Project/results/model_comparison.csv`

Current table values:

- Random Forest: RMSE 351, MAE 201, R2 0.866
- XGBoost: RMSE 465, MAE 299, R2 0.765

## 7.6 Inference Testing (`06_Model_Testing.ipynb`)

Behavior:

- Loads RF model + encoders + scaler.
- Constructs single-row input.
- Reorders columns to model training order.
- Encodes categorical values.
- Scales and predicts yield.

Observed runtime issue:

- Label encoder raises unseen-label error for `Rice` due case mismatch against trained category values (e.g., `rice`).

## 8. Yield Data and Artifacts

### 8.1 Processed Data

Primary cleaned file:

- `ML1/ML_Yield_Project/data/processed/cleaned_dataset.csv`

Observed shape:

- 50,765 rows x 15 columns

Columns:

- `Year`
- `State Name`
- `Dist Name`
- `Crop`
- `Area_ha`
- `Yield_kg_per_ha`
- `N_req_kg_per_ha`
- `P_req_kg_per_ha`
- `K_req_kg_per_ha`
- `Temperature_C`
- `Humidity_%`
- `pH`
- `Rainfall_mm`
- `Wind_Speed_m_s`
- `Solar_Radiation_MJ_m2_day`

Distinct value highlights:

- Crop: 4 categories
- State Name: 20 categories
- Dist Name: 311 categories

### 8.2 Serialized Feature Arrays

File:

- `ML1/ML_Yield_Project/data/processed/.npz`

Stored tensors:

- `X_train`: (40612, 14)
- `X_test`: (10153, 14)
- `y_train`: (40612,)
- `y_test`: (10153,)

### 8.3 Yield Models and Preprocessors

Directory:

- `ML1/ML_Yield_Project/models/`

Includes:

- `rf_model.pkl`
- `xgb_model.pkl`
- `crop_encoder.pkl`
- `state_encoder.pkl`
- `district_encoder.pkl`
- `scaler.pkl`

Size observations:

- `rf_model.pkl`: very large (~216 MB)
- `xgb_model.pkl`: compact (~1 MB)

Duplication note:

- `ML1/models/rf_model.pkl` appears to be a duplicate copy of the RF model.

## 9. File-Type Inventory Snapshot (ML1)

Total files under `ML1`: 1600

Top extensions:

- `.jpg`: 747
- `.heic`: 687
- `.png`: 74
- `.jpeg`: 58
- `.ipynb`: 10
- `.webp`: 9
- `.pkl`: 7
- `.keras`: 5
- `.csv`: 2

Interpretation:

- Workspace is heavily image-centric.
- Mix of image formats implies heterogeneous acquisition sources.

## 10. Operational Dependencies and Runtime Assumptions

The notebooks assume these Python libraries are available:

- tensorflow / keras
- numpy
- pandas
- scikit-learn
- xgboost
- opencv-python
- matplotlib
- seaborn
- kagglehub

Runtime assumptions:

- Many cells assume Google Colab and Google Drive mounting.
- Absolute Drive paths are used in several notebook sections.

## 11. Current Strengths

- End-to-end disease pipeline exists from dataset split to saved models to inference testing.
- End-to-end yield pipeline exists from data cleaning to training to model comparison and test inference.
- Artifacts are already generated and reusable.
- Multi-crop disease architecture is consistent and relatively modular at the notebook level.

## 12. Known Issues and Technical Debt

1. Non-deterministic image splits due no fixed random seed.
2. Class naming typo (`Bacrerial_Wilt`) embedded in dataset and labels.
3. Source/split dataset mismatch complicates reproducibility and provenance.
4. Manual crop selection and repeated training loops increase human error risk.
5. Hidden `.npz` filename reduces clarity and portability.
6. Case-sensitive categorical encoding causes inference failures (`Rice` vs `rice`).
7. Duplicate large RF model file wastes storage.
8. Some notebook cells appear duplicated or include stale/hardcoded result blocks.

## 13. Reproducibility and MLOps Readiness Assessment

Current maturity:

- Prototype to pre-production research stage.

What exists:

- Trained models and preprocessing artifacts.
- Basic evaluation flow.
- Initial web integration inference prototype.

Missing for production:

- Unified pipeline scripts (non-notebook batch entry points)
- Versioned dataset registry and exact lineage tracking
- Deterministic split generation with seed control
- Schema validation for tabular inputs
- Robust artifact versioning and experiment tracking
- CI for training/evaluation checks
- Service packaging and API contract definition

## 14. Recommended Immediate Improvements

1. Replace all notebook hardcoded paths with config-driven relative paths.
2. Set global seeds for Python, NumPy, and TensorFlow during split/training.
3. Normalize label naming conventions (fix typos and case handling).
4. Rename hidden `.npz` artifact to explicit filename (for example `train_test_arrays.npz`).
5. Add input sanitation to yield inference (`strip`, lowercase, controlled vocab).
6. Remove duplicate model artifacts and define canonical model directory.
7. Convert key notebook flows into reproducible Python scripts or pipeline runners.
8. Add a single metrics registry (CSV/JSON) generated directly from executed evaluations.

## 15. Runbook (As Currently Implemented)

### 15.1 Disease Model Workflow

1. Execute `ML1/notebooks/01_Data_Preparation.ipynb` to generate split directories.
2. Set `CROP_NAME` and run `ML1/notebooks/02_MobileNetV2_Training.ipynb` for each crop.
3. Run `ML1/notebooks/03_model_evalution.ipynb` for test metrics and confusion matrix.
4. Use `ML1/notebooks/04_web_integration_test.ipynb` for upload-based inference checks.

### 15.2 Yield Model Workflow

1. Run `ML1/ML_Yield_Project/notebooks/01_Data_Preprocessing.ipynb` to clean data.
2. Run `ML1/ML_Yield_Project/notebooks/02_Feature_Engineering.ipynb` to create encoders/scaler and arrays.
3. Train RF using `ML1/ML_Yield_Project/notebooks/03_Model_Training_RF.ipynb`.
4. Train XGBoost using `ML1/ML_Yield_Project/notebooks/04_Model_Training_XGBoost.ipynb`.
5. Write comparison table in `ML1/ML_Yield_Project/notebooks/05_Model_Comparison.ipynb`.
6. Test single-row inference in `ML1/ML_Yield_Project/notebooks/06_Model_Testing.ipynb`.

## 16. Conclusion

ML1 is a substantial hackathon-grade applied ML workspace that already demonstrates two meaningful agritech capabilities: disease diagnosis from leaf images and crop yield estimation from tabular features. The core models and artifacts are present and usable. The main next step is not model invention, but system hardening: reproducibility, consistent naming, artifact hygiene, and migration from exploratory notebooks to repeatable pipeline code.
