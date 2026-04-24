"""
Aegis Fraud Detection System - Preprocessing Module
Part of a thesis-grade ML pipeline for financial fraud detection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import joblib
import os

def load_data(file_path):
    """
    Loads the dataset and prints basic exploration metrics.
    """
    print(f"[PHASE 1] Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Null Values: {df.isnull().sum().max()}")
    
    class_dist = df['Class'].value_counts(normalize=True) * 100
    print(f"Class Distribution:\n{class_dist}")
    
    return df

def preprocess_pipeline(df, random_state=42):
    """
    Complete preprocessing pipeline: Scaling, Splitting, and Resampling.
    """
    print("[PHASE 1] Starting Preprocessing Pipeline...")
    
    # 1. Scaling Time and Amount
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # 2. Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    print(f"Train/Test Split complete. Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # 3. Handling Imbalance
    datasets = {
        'original': (X_train, y_train),
        'smote': SMOTE(random_state=random_state).fit_resample(X_train, y_train),
        'adasyn': ADASYN(random_state=random_state).fit_resample(X_train, y_train),
        'undersample': RandomUnderSampler(random_state=random_state).fit_resample(X_train, y_train)
    }
    
    for name, (X_res, y_res) in datasets.items():
        print(f"Resampling Technique: {name.upper()} - New Shape: {X_res.shape}")
        
    return datasets, (X_test, y_test)

if __name__ == "__main__":
    DATA_PATH = "fraud_detection/data/creditcard.csv"
    if os.path.exists(DATA_PATH):
        df = load_data(DATA_PATH)
        datasets, test_set = preprocess_pipeline(df)
        
        # Save preprocessed data for training phase
        os.makedirs("fraud_detection/results/data", exist_ok=True)
        joblib.dump((datasets, test_set), "fraud_detection/results/data/preprocessed_data.pkl")
        print("[PHASE 1] Preprocessing complete. Data saved.")
    else:
        print(f"Error: {DATA_PATH} not found.")
