"""
Aegis Fraud Detection System - Evaluation Module
Generates comprehensive metrics and visualization plots for thesis presentation.
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, f1_score, 
    matthews_corrcoef, confusion_matrix, classification_report,
    auc, roc_curve
)
import os

def evaluate_models(models_tuple, test_set):
    """
    Computes metrics for all model combinations and saves heatmaps for the top 3.
    """
    print("[PHASE 3] Starting Comprehensive Evaluation...")
    baseline_models, advanced_models = models_tuple
    X_test, y_test = test_set
    
    results_list = []
    os.makedirs("fraud_detection/results/plots", exist_ok=True)
    os.makedirs("fraud_detection/results/metrics", exist_ok=True)
    
    # Storage for models and their predictions to avoid re-calculation for top 3
    eval_cache = {}

    # Evaluate Baseline
    for dataset_name, models in baseline_models.items():
        for model_name, model in models.items():
            identifier = f"{model_name}_{dataset_name}"
            metrics = run_evaluation(model, X_test, y_test, identifier, results_list)
            eval_cache[identifier] = (model, metrics)
            
    # Evaluate Advanced
    for full_name, model in advanced_models.items():
        metrics = run_evaluation(model, X_test, y_test, full_name, results_list)
        eval_cache[full_name] = (model, metrics)
        
    results_df = pd.DataFrame(results_list)
    results_df.to_csv("fraud_detection/results/metrics/master_comparison.csv", index=False)
    
    # Identify Top 3 Models based on MCC (or AUC-ROC if MCC is equal)
    top_3 = results_df.sort_values(by=['MCC', 'AUC-ROC'], ascending=False).head(3)
    
    print("\n[TOP 3 PERFORMING MODELS]")
    print(top_3[['Model', 'AUC-ROC', 'MCC']])
    
    # Generate enhanced heatmaps for Top 3
    for idx, row in top_3.iterrows():
        model_name = row['Model']
        model, _ = eval_cache[model_name]
        save_enhanced_heatmap(model, X_test, y_test, model_name)

    print("[PHASE 3] Evaluation complete. Metrics saved to results/metrics/master_comparison.csv")
    return results_df

def run_evaluation(model, X_test, y_test, identifier, results_list):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    auc_roc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    metrics = {
        'Model': identifier,
        'AUC-ROC': auc_roc,
        'F1-Score': f1,
        'MCC': mcc
    }
    results_list.append(metrics)
    
    # Standard CM for every model
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'CM: {identifier}')
    plt.savefig(f'fraud_detection/results/plots/cm_{identifier}.png')
    plt.close()
    
    return metrics

def save_enhanced_heatmap(model, X_test, y_test, identifier):
    """
    Saves a high-resolution, annotated heatmap for top performers.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
    
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    
    sns.heatmap(cm, annot=labels, fmt='', cmap='vlag', cbar=True)
    plt.title(f'BEST PERFORMER: {identifier}\nConfusion Matrix Diagnostic', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    save_path = f'fraud_detection/results/plots/BEST_MODEL_CM_{identifier}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Enhanced diagnostic heatmap saved for {identifier} at {save_path}")

if __name__ == "__main__":
    MODELS_PATH = "fraud_detection/models/trained_models.pkl"
    DATA_PATH = "fraud_detection/results/data/preprocessed_data.pkl"
    
    if os.path.exists(MODELS_PATH) and os.path.exists(DATA_PATH):
        models = joblib.load(MODELS_PATH)
        _, test_set = joblib.load(DATA_PATH)
        evaluate_models(models, test_set)
    else:
        print("Required files not found.")
