import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from preprocess import FeatureEngineer, get_preprocessor

def train_model(data_path, model_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    if 'Exited' not in df.columns:
        raise ValueError("Target column 'Exited' not found in dataset.")
        
    X = df.drop(columns=['Exited'])
    y = df['Exited']
    
    # Stratified split to maintain class ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create Full Pipeline
    pipeline = Pipeline([
        ('feature_engineer', FeatureEngineer()),
        ('preprocessor', get_preprocessor()),
        ('classifier', RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, class_weight='balanced'))
    ])
    
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(base_dir, 'data', 'churn.csv')
    MODEL_PATH = os.path.join(base_dir, 'model', 'churn_model.pkl')
    train_model(DATA_PATH, MODEL_PATH)
