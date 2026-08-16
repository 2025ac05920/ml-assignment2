# ML Assignment 2 — Wine Quality Classification

## a. Problem Statement

Build and compare **5 classification models** on a single dataset, evaluate each using 6 metrics (Accuracy, AUC, Precision, Recall, F1, MCC), and deploy an interactive Streamlit web app that lets a user upload test data, pick a model, and view results.

The dataset chosen is the **Wine Quality (Red)** dataset from UCI/Kaggle. The task is framed as a **binary classification** problem: predict whether a wine is "good" (quality ≥ 6) or "bad" (quality < 6) based on its physicochemical properties.

---

## b. Dataset Description

**Source:** UCI Machine Learning Repository / Kaggle  
**Link:** <https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009>  
**Instances:** 1599  
**Features:** 11 physicochemical input variables + 1 output (quality)

| # | Feature | Description | Unit |
| --- | --------- | ------------- | ------ |
| 1 | fixed acidity | most acids involved with wine | g/dm³ |
| 2 | volatile acidity | acetic acid in wine | g/dm³ |
| 3 | citric acid | citric acid content | g/dm³ |
| 4 | residual sugar | sugar remaining after fermentation | g/dm³ |
| 5 | chlorides | amount of salt | g/dm³ |
| 6 | free sulfur dioxide | free SO₂ | mg/dm³ |
| 7 | total sulfur dioxide | total SO₂ | mg/dm³ |
| 8 | density | density of wine | g/cm³ |
| 9 | pH | pH level | 0–14 |
| 10 | sulphates | potassium sulphate | g/dm³ |
| 11 | alcohol | alcohol content | % vol |
| 12 | quality | output (0–10) → converted to binary target | — |

**Target:** `quality >= 6 → 1 (good), else 0 (bad)`

This satisfies the assignment constraints: **12 features** (11 input + derived target) and **1599 instances** (> 500).

---

## c. GitHub Repository Link

**BITS ID:** 2025ac05920  
**Student:** Niraj Kumar Rai

GitHub Repo: [https://github.com/2025ac05920/ml-assignment2](https://github.com/2025ac05920/ml-assignment2)

---

## d. Models Used & Comparison Table

### Evaluation Metrics Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.7406 | 0.8242 | 0.7683 | 0.7368 | 0.7522 | 0.4808 |
| Decision Tree | 0.7312 | 0.7830 | 0.7545 | 0.7368 | 0.7456 | 0.4610 |
| kNN | 0.7406 | 0.8117 | 0.7588 | 0.7544 | 0.7566 | 0.4790 |
| Naive Bayes | 0.7250 | 0.7909 | 0.7785 | 0.6784 | 0.7250 | 0.4569 |
| Random Forest (Ensemble) | 0.8063 | 0.9018 | 0.8344 | 0.7953 | 0.8144 | 0.6128 |

### Observations Table

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Linear model; performs reasonably if classes are linearly separable. Sensitive to feature scaling. |
| Decision Tree | Captures non-linear relationships; may overfit if depth is too high. Pruned with max_depth=5. |
| kNN | Instance-based; sensitive to feature scaling and choice of k. k=5 is a common default. |
| Naive Bayes | Assumes feature independence; fast but may underperform if features are correlated. |
| Random Forest (Ensemble) | Combines many decision trees; typically the strongest performer with good generalization. |

### Overall Winner for this dataset?

> **Random Forest** — achieves the highest F1 (0.8144), AUC (0.9018), and MCC (0.6128) on the Wine Quality dataset. Its ensemble nature (100 decision trees averaged) captures non-linear feature interactions without overfitting, outperforming all other models by a clear margin.

---

## Project Structure

```
ml-assignment2-wine-quality/
│
├── app.py                  # Streamlit web app
├── requirements.txt
├── README.md
├── .gitignore
├── test_data.csv           # 20% hold-out test set (auto-generated)
├── data/
│   └── winequality-red.csv   # (download from Kaggle)
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest.joblib
    ├── scaler.joblib
    ├── metrics.json
    └── features.json
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download dataset

Download `winequality-red.csv` from [Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) and place it in `data/`.

### 3. Train models

Open and run the notebook `model/model_training.ipynb` in Jupyter or VS Code:

```bash
jupyter notebook model/model_training.ipynb
```

Run all cells from top to bottom. This generates:

- Saved models in `model/` (`.joblib` files)
- `test_data.csv` (the hold-out test set)
- `model/metrics.json` (all evaluation metrics)

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

Upload `test_data.csv` in the app to see predictions and metrics.

---

## Streamlit App Features

- ✅ **Dataset upload (CSV)** — upload test data
- ✅ **Model selection dropdown** — choose from 5 models
- ✅ **Evaluation metrics display** — Accuracy, AUC, Precision, Recall, F1, MCC
- ✅ **Confusion matrix** — visual heatmap
- ✅ **Classification report** — full text report

---

## Streamlit App Link

> **Replace this with your deployed app URL after deploying.**
>
> Live App: `https://2025ac05920-WQC.streamlit.app`
