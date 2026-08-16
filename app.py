"""
Streamlit app for ML Assignment 2 — Wine Quality Classification.

Features:
  a. Dataset upload option (CSV) — upload test data
  b. Model selection dropdown
  c. Display of evaluation metrics
  d. Confusion matrix + classification report

Run locally:
  streamlit run app.py
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

MODEL_DIR = "model"

# Models that use scaled features
SCALED_MODELS = {"Logistic Regression", "kNN"}


@st.cache_resource
def load_metrics():
    path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_resource
def load_features():
    path = os.path.join(MODEL_DIR, "features.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_resource
def load_model(model_name):
    safe_name = model_name.replace(" ", "_").lower()
    path = os.path.join(MODEL_DIR, f"{safe_name}.joblib")
    return joblib.load(path)


@st.cache_resource
def load_scaler():
    path = os.path.join(MODEL_DIR, "scaler.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AUC": round(roc_auc_score(y_true, y_proba), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1": round(f1_score(y_true, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4),
    }


def main():
    st.set_page_config(page_title="ML Assignment 2 — Wine Quality Classifier", page_icon="🍷")
    st.title("🍷 Wine Quality Classification")
    st.write("Upload your test data CSV and select a model to see predictions and metrics.")

    # --- Sidebar header ---
    st.sidebar.markdown("# ML Assignment 2")
    st.sidebar.markdown("**Wine Quality Classification**")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Developed by:**")
    st.sidebar.markdown("Niraj Kumar Rai")
    st.sidebar.markdown("2025ac05920")
    st.sidebar.markdown("M.Tech (AIML)")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**GitHub Repository:**")
    st.sidebar.markdown("[2025ac05920/ml-assignment2](https://github.com/2025ac05920/ml-assignment2)")
    st.sidebar.markdown("---")

    # --- Model selection dropdown ---
    model_names = [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest",
    ]
    selected_model = st.sidebar.selectbox("Select Model", model_names)

    # --- File upload ---
    uploaded_file = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"])

    # Show saved metrics from training
    saved_metrics = load_metrics()
    if saved_metrics:
        st.sidebar.markdown("### Saved Metrics (from training)")
        if selected_model in saved_metrics:
            sm = saved_metrics[selected_model]
            for k, v in sm.items():
                st.sidebar.write(f"**{k}**: {v}")

    if uploaded_file is not None:
        try:
            # Try comma first, then semicolon (UCI wine format)
            try:
                df = pd.read_csv(uploaded_file)
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=";")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return

        st.write("### Uploaded Data")
        st.write(f"Shape: {df.shape}")
        st.dataframe(df.head())

        features = load_features()
        if features is None:
            st.error("Feature list not found. Run train_models.py first.")
            return

        # Check required columns
        missing = [f for f in features if f not in df.columns]
        if missing:
            st.error(f"Missing required feature columns: {missing}")
            st.info(f"Expected columns: {features}")
            return

        if "target" not in df.columns:
            st.error("CSV must contain a 'target' column for evaluation.")
            return

        X = df[features]
        y_true = df["target"]

        # Load model and predict
        model = load_model(selected_model)
        scaler = load_scaler()

        if selected_model in SCALED_MODELS and scaler is not None:
            X_input = scaler.transform(X)
            X_input = pd.DataFrame(X_input, columns=X.columns)
        else:
            X_input = X

        y_pred = model.predict(X_input)
        y_proba = model.predict_proba(X_input)[:, 1]

        # Compute metrics
        metrics = compute_metrics(y_true, y_pred, y_proba)

        # --- Display metrics ---
        st.write(f"### Results: {selected_model}")
        cols = st.columns(3)
        for i, (k, v) in enumerate(metrics.items()):
            with cols[i % 3]:
                st.metric(label=k, value=v)

        # --- Confusion matrix ---
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Bad", "Good"],
            yticklabels=["Bad", "Good"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # --- Classification report ---
        st.write("### Classification Report")
        report = classification_report(y_true, y_pred, target_names=["Bad", "Good"])
        st.text(report)

    else:
        st.info("Upload a CSV file from the sidebar to get started.")
        st.write("The CSV should contain the feature columns and a 'target' column.")
        st.write("You can use `test_data.csv` from this repo as a sample.")


if __name__ == "__main__":
    main()
