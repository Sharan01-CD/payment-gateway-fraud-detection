import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

def bootstrap():
    print("Bootstrapping Aegis Demo Assets...")
    
    # 1. Create Directories
    os.makedirs("fraud_detection/data", exist_ok=True)
    os.makedirs("fraud_detection/models", exist_ok=True)
    os.makedirs("fraud_detection/results/data", exist_ok=True)
    
    # 2. Create Dummy Data
    print("Generating synthetic creditcard.csv...")
    n_samples = 1000
    n_features = 30
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    columns = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
    df = pd.DataFrame(X, columns=columns)
    df['Class'] = y
    df.to_csv("fraud_detection/data/creditcard.csv", index=False)
    
    # 3. Create Dummy Preprocessed Data
    print("Generating preprocessed_data.pkl...")
    X_train = df.drop('Class', axis=1).iloc[:800]
    y_train = df['Class'].iloc[:800]
    X_test = df.drop('Class', axis=1).iloc[800:]
    y_test = df['Class'].iloc[800:]
    
    datasets = {
        'original': (X_train, y_train),
        'smote': (X_train, y_train) # Mocking smote
    }
    test_set = (X_test, y_test)
    joblib.dump((datasets, test_set), "fraud_detection/results/data/preprocessed_data.pkl")
    
    # 4. Create Dummy Models
    print("Generating trained_models.pkl...")
    lr = LogisticRegression().fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)
    mlp = MLPClassifier(max_iter=10).fit(X_train, y_train)
    
    xgb_model = xgb.XGBClassifier(n_estimators=5).fit(X_train, y_train)
    
    baseline_models = {
        'original': {
            'logistic_regression': lr,
            'random_forest': rf,
            'mlp': mlp
        }
    }
    advanced_models = {
        'xgb_smote': xgb_model
    }
    
    joblib.dump((baseline_models, advanced_models), "fraud_detection/models/trained_models.pkl")
    
    print("Bootstrap complete. All folders and dummy assets are ready.")

if __name__ == "__main__":
    bootstrap()
