import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Labels from app.py
DISEASE_LABELS = {
    "Brinjal": ["Bacterial Wilt", "Cercospora Leaf Spot", "Healthy", "Mosaic", "Phomopsis Leaf Blight"],
    "Castor": ["Alternaria Leaf Blight", "Bacterial Leaf Blight", "Cercospora Leaf Spot", "Healthy", "Leaf Curv Virus"],
    "Cumin": ["Alternaria Blight", "Healthy", "Wilt"],
    "Guava": ["Anthracnose", "Bacterial Blight", "Healthy", "Red Rust", "Wilt"],
    "Papaya": ["Healthy", "Leaf Spot", "Powdery Mildew", "Ring Spot Virus"]
}

def verify_crop(crop_name):
    model_path = os.path.join(MODELS_DIR, f"{crop_name}_model.keras")
    if not os.path.exists(model_path):
        print(f"Error: Model for {crop_name} not found at {model_path}")
        return

    print(f"\nVerifying {crop_name} disease classification model...")
    try:
        model = load_model(model_path)
        print(f"Model for {crop_name} loaded successfully!")
        
        # Try to find an image in the dataset to test
        crop_data_dir = os.path.join(DATASET_DIR, crop_name)
        if os.path.exists(crop_data_dir):
            # Find the first image in the first subdirectory
            for disease_dir in os.listdir(crop_data_dir):
                disease_path = os.path.join(crop_data_dir, disease_dir)
                if os.path.isdir(disease_path):
                    images = [f for f in os.listdir(disease_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if images:
                        test_img_path = os.path.join(disease_path, images[0])
                        print(f"Testing with image: {test_img_path}")
                        
                        # Preprocess
                        img = cv2.imread(test_img_path)
                        img_resized = cv2.resize(img, (224, 224))
                        img_normalized = img_resized / 255.0
                        img_batch = np.expand_dims(img_normalized, axis=0)
                        
                        # Predict
                        preds = model.predict(img_batch, verbose=0)
                        class_idx = np.argmax(preds[0])
                        confidence = preds[0][class_idx]
                        
                        label = DISEASE_LABELS[crop_name][class_idx]
                        print(f"Diagnosis: {label}")
                        print(f"Confidence: {confidence*100:.2f}%")
                        break
        else:
            # Dummy prediction with random noise
            print("Dataset not found, running dummy prediction with random noise...")
            dummy_img = np.random.rand(1, 224, 224, 3).astype('float32')
            preds = model.predict(dummy_img, verbose=0)
            print("Model prediction executed successfully!")
            
    except Exception as e:
        print(f"Error during verification: {e}")

# Verify one model as a representative
verify_crop("Brinjal")
