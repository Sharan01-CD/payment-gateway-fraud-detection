"""
Aegis Fraud Detection System - Training Module
Trains multiple models using Optuna for hyperparameter tuning.
"""

import joblib
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
import os

def train_baseline_models(datasets):
    """
    Trains baseline models: Logistic Regression, Random Forest, MLP, Isolation Forest.
    """
    print("[PHASE 2] Training Baseline Models...")
    results = {}
    
    for name, (X_train, y_train) in datasets.items():
        print(f"Training on {name} dataset...")
        
        # Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced' if name == 'original' else None)
        lr.fit(X_train, y_train)
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # MLP
        mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        mlp.fit(X_train, y_train)
        
        results[name] = {
            'logistic_regression': lr,
            'random_forest': rf,
            'mlp': mlp
        }
    
    return results

def optimize_xgb(X_train, y_train, X_val, y_val):
    def objective(trial):
        param = {
            'verbosity': 0,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'eta': trial.suggest_float('eta', 1e-3, 0.1, log=True),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        }
        model = xgb.XGBClassifier(**param)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    return study.best_params

def train_advanced_models(datasets, test_set):
    """
    Hyperparameter tuning for XGBoost and LightGBM.
    """
    print("[PHASE 2] Starting Hyperparameter Tuning with Optuna...")
    X_test, y_test = test_set
    advanced_results = {}
    
    for name, (X_train, y_train) in datasets.items():
        print(f"Optimizing XGBoost for {name}...")
        best_params = optimize_xgb(X_train, y_train, X_test, y_test)
        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train)
        advanced_results[f"xgb_{name}"] = model
        
    return advanced_results

if __name__ == "__main__":
    DATA_PATH = "fraud_detection/results/data/preprocessed_data.pkl"
    if os.path.exists(DATA_PATH):
        datasets, test_set = joblib.load(DATA_PATH)
        baseline_models = train_baseline_models(datasets)
        advanced_models = train_advanced_models(datasets, test_set)
        
        os.makedirs("fraud_detection/models", exist_ok=True)
        joblib.dump((baseline_models, advanced_models), "fraud_detection/models/trained_models.pkl")
        print("[PHASE 2] Training complete. Models saved.")
    else:
        print("Data not found. Run preprocess.py first.")
