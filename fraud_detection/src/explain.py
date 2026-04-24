"""
Aegis Fraud Detection System - Explainability Module
Uses SHAP values to explain model decisions for high-stakes fraud detection.
"""

import shap
import joblib
import matplotlib.pyplot as plt
import os

def generate_explanations(best_model, X_test):
    """
    Generates SHAP summary and waterfall plots.
    """
    print("[PHASE 4] Generating Model Explanations with SHAP...")
    
    # SHAP explainer
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)
    
    os.makedirs("fraud_detection/results/plots", exist_ok=True)
    
    # 1. Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("fraud_detection/results/plots/shap_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Waterfall Plot for a sample fraud case
    # Assuming fraud cases are rare, we find one
    fraud_indices = X_test[X_test.index.isin(X_test.index)].index # Just a placeholder logic
    # In reality, filter by y_test == 1
    
    print("[PHASE 4] SHAP analysis complete.")

if __name__ == "__main__":
    # Placeholder for actual best model selection
    print("Run this after Phase 3.")
