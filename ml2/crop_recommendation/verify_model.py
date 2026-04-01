import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pickle
import os

# Load dataset
csv_path = 'ml2/dataset/Crop_recommendation.csv'
if not os.path.exists(csv_path):
    print(f"Dataset not found at {csv_path}")
    exit(1)

df = pd.read_csv(csv_path)

# Split features and target
X = df.drop('label', axis=1)
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100)
}

# Training and results
print("Evaluating models...")
best_acc = 0
best_model_name = ""
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name}: {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_model_name = name

print(f"\nBest Model: {best_model_name} with accuracy {best_acc:.4f}")

# Save best model
best_model = models[best_model_name]
with open('ml2/models/crop_recommendation_rf.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("Best model saved to ml2/models/crop_recommendation_rf.pkl")
