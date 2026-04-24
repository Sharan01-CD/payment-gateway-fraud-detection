import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def build_simple_model(csv_path):
    """
    Trains a more robust Random Forest model using all PCA features (V1-V28) and Amount.
    """
    print(f"Loading data from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please ensure the dataset is in the correct location.")
        return

    df = pd.read_csv(csv_path)
    
    # Scale Amount
    df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    
    # Use V1-V28 and Amount
    v_features = [f'V{i}' for i in range(1, 29)]
    features_to_use = v_features + ['Amount']
    
    X = df[features_to_use]
    y = df['Class']
    
    # Stratified split to handle imbalance better in the training set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training Robust Random Forest on {len(features_to_use)} features...")
    # Increase estimators and use balanced class weight for better performance on minority class
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=12, 
        random_state=42, 
        class_weight='balanced_subsample'
    )
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'robust_fraud_model.pkl')
    print("Model saved to robust_fraud_model.pkl")

def predict(amount, v_values):
    """
    Predicts using the robust model.
    v_values: dictionary or list of values for V1-V28
    """
    model_path = 'robust_fraud_model.pkl'
    if not os.path.exists(model_path):
        return "MODEL NOT TRAINED - Run build_simple_model first"

    model = joblib.load(model_path)
    
    # Ensure v_values is flattened correctly for the model
    # Expecting 29 features: V1-V28, Amount
    if isinstance(v_values, dict):
        ordered_v = [v_values.get(f'V{i}', 0.0) for i in range(1, 29)]
    else:
        ordered_v = v_values # Assume it's already a list of 28 values
        
    features = [ordered_v + [amount]]
    prediction = model.predict(features)[0]
    return "FRAUD" if prediction == 1 else "LEGIT"

if __name__ == "__main__":
    # Example usage:
    # build_simple_model('fraud_detection/data/creditcard.csv')
    pass
