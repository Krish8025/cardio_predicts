import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train():
    # Define paths
    # Data is now in the same directory (backend) or needing adjustment?
    # User moved files to backend. So cardio_train_cleaned.csv is in backend/.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "cardio_train_cleaned.csv")
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Features and Target
    X = df.drop('cardio', axis=1)
    y = df['cardio']

    # Split
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Gradient Boosting
    print("Training Gradient Boosting Classifier...")
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Save
    print("Saving model and scaler...")
    joblib.dump(model, os.path.join(base_dir, 'cardio_model_gb.pkl'))
    joblib.dump(scaler, os.path.join(base_dir, 'scaler.pkl'))
    print("Done.")

if __name__ == "__main__":
    train()
