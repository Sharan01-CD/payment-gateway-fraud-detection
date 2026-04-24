# 🛡️ Aegis Fraud Detection Engine

A high-precision Machine Learning framework designed to detect fraudulent credit card transactions in real-time. Built with **XGBoost** and **SMOTE** to handle extreme class imbalance.

## 📂 Project Structure
- `app_streamlit.py`: Main dashboard entry point.
- `requirements.txt`: Python dependencies.
- `bootstrap_demo.py`: Utility to generate synthetic assets for trial.
- `fraud_detection/`: Core ML pipeline directory.
  - `src/`: Training and processing scripts.
  - `models/`: Where `.pkl` files are stored.
  - `data/`: Dataset storage.

## 🚀 Getting Started
1. **Local Setup**:
   ```bash
   pip install -r requirements.txt
   streamlit run app_streamlit.py
   ```
2. **Initialization**: On first run, use the sidebar button inside the app to generate dummy models if you haven't trained real ones yet.

## 📊 Methodology
Aegis addresses class imbalance (0.17% fraud) by synthesizing minority class samples using **SMOTE**, followed by gradient boosting via **XGBoost** to refine decision boundaries.
