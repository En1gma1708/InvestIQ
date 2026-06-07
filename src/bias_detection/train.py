import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from xgboost import XGBClassifier

def train_models():
    data_path = 'C:/Users/Sahil Sharma/Projects/InvestIQ/data/behavioral_profiles.csv'
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}. Please run data_generator.py first.")
        return
        
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    os.makedirs('C:/Users/Sahil Sharma/Projects/InvestIQ/models', exist_ok=True)
    joblib.dump(scaler, 'C:/Users/Sahil Sharma/Projects/InvestIQ/models/scaler.pkl')
    print("StandardScaler saved to models/scaler.pkl")
    
    # 1. Random Forest Classifier
    print("\nTraining Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train_scaled, y_train)
    
    rf_preds = rf_model.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds, average='weighted')
    
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    print(f"Random Forest F1 (Weighted): {rf_f1:.4f}")
    
    # 2. XGBoost Classifier
    print("\nTraining XGBoost Classifier...")
    xgb_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    
    xgb_preds = xgb_model.predict(X_test_scaled)
    xgb_acc = accuracy_score(y_test, xgb_preds)
    xgb_f1 = f1_score(y_test, xgb_preds, average='weighted')
    
    print(f"XGBoost Accuracy: {xgb_acc:.4f}")
    print(f"XGBoost F1 (Weighted): {xgb_f1:.4f}")
    
    # Save the best model
    best_model = None
    best_name = ""
    if xgb_f1 > rf_f1:
        best_model = xgb_model
        best_name = "XGBoost"
    else:
        best_model = rf_model
        best_name = "Random Forest"
        
    model_path = 'C:/Users/Sahil Sharma/Projects/InvestIQ/models/bias_detector_model.pkl'
    joblib.dump(best_model, model_path)
    print(f"\nBest model ({best_name}) saved to {model_path}")
    
    # Detailed report for best model
    best_preds = best_model.predict(X_test_scaled)
    target_names = [
        'Rational (No Bias)', 
        'Loss Aversion', 
        'FOMO/Herd Mentality', 
        'Overconfidence', 
        'Recency Bias'
    ]
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, best_preds, target_names=target_names))
    
    # Save feature importances
    importances = best_model.feature_importances_
    features = X.columns
    feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)
    
    results_dir = 'C:/Users/Sahil Sharma/Projects/InvestIQ/results'
    os.makedirs(results_dir, exist_ok=True)
    feat_imp_df.to_csv(f'{results_dir}/feature_importances.csv', index=False)
    print(f"Feature importances saved to {results_dir}/feature_importances.csv")
    print(feat_imp_df)

if __name__ == '__main__':
    train_models()
