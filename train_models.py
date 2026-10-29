"""
Diabetes Prediction - Model Training Pipeline
================================================
College ML project inspired by the "Diabetes Prediction: Transforming
Healthcare with Machine Learning" brief.

What this script does (matches the pipeline in the brief):
  1. Data Collection      -> load Pima Indians Diabetes dataset
  2. Preprocessing        -> fix biologically-impossible zeros, scale features
  3. Handle imbalance     -> SMOTE oversampling on the training set only
  4. Model Training       -> Logistic Regression, Random Forest, AdaBoost,
                             and a Stacking ensemble of all three
  5. Evaluation           -> stratified 5-fold CV, AUC-ROC, precision/recall,
                             confusion matrix
  6. Explainability       -> SHAP summary + feature importance plots
  7. Deployment prep      -> saves the best model + scaler for the Streamlit app

Run with:  python train_models.py
"""

import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

DATA_PATH = "data/diabetes.csv"
MODEL_DIR = "saved_models"
REPORT_DIR = "reports"

FEATURE_COLS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
TARGET_COL = "Outcome"

# Columns where 0 is not a physiologically real value -> treat as missing
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_and_clean_data():
    """Step 1-2: Data Collection + Preprocessing."""
    df = pd.read_csv(DATA_PATH)

    # Replace impossible zeros with NaN, then impute with the median
    # (a simple, defensible strategy to call out and explain to your professor)
    for col in ZERO_AS_MISSING:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

    return df


def build_models():
    """Step 4: define the models used, mirroring the brief's 'Algorithms Used' slide."""
    log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=RANDOM_STATE)
    ada = AdaBoostClassifier(n_estimators=150, random_state=RANDOM_STATE)

    # Stacking ensemble: combine the three base learners with a logistic
    # regression meta-learner (the "Ensembles" slide -> AdaBoost/Stacking)
    stack = StackingClassifier(
        estimators=[("log_reg", log_reg), ("rf", rf), ("ada", ada)],
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        cv=5,
    )

    return {
        "Logistic Regression": log_reg,
        "Random Forest": rf,
        "AdaBoost": ada,
        "Stacking Ensemble": stack,
    }


def evaluate_model(name, model, X_test, y_test, cv_scores):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "cv_auc_mean": round(cv_scores.mean(), 4),
        "cv_auc_std": round(cv_scores.std(), 4),
    }
    print(f"\n--- {name} ---")
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))
    return metrics, y_proba, y_pred


def plot_roc_curves(results, X_test, y_test):
    plt.figure(figsize=(7, 6))
    for name, (model, _, y_proba) in results.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Diabetes Risk Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/roc_curves.png", dpi=150)
    plt.close()


def plot_precision_recall(results, X_test, y_test):
    plt.figure(figsize=(7, 6))
    for name, (model, _, y_proba) in results.items():
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        plt.plot(rec, prec, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves - Diabetes Risk Models")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/precision_recall_curves.png", dpi=150)
    plt.close()


def plot_confusion_matrix(name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="RdPu",
                xticklabels=["No Diabetes", "Diabetes"],
                yticklabels=["No Diabetes", "Diabetes"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(f"{REPORT_DIR}/confusion_matrix_{safe_name}.png", dpi=150)
    plt.close()


def plot_shap_summary(model, X_train_df, X_test_df):
    """Explainability slide: SHAP values + feature importance for clinician trust."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_df)

    # Different shap versions return this differently for binary classifiers:
    # - a list [class0_values, class1_values], or
    # - a single array of shape (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    plt.figure()
    shap.summary_plot(sv, X_test_df, show=False)
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(sv, X_test_df, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/shap_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    import os
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("Loading and cleaning data...")
    df = load_and_clean_data()
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    print(f"Dataset shape: {df.shape} | Positive rate: {y.mean():.2%}")

    # Train/test split BEFORE any resampling to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Scale features (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 3: fix class imbalance with SMOTE (train set only)
    print("\nApplying SMOTE to balance classes on the training set...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
    print(f"After SMOTE:  {pd.Series(y_train_bal).value_counts().to_dict()}")

    models = build_models()
    results = {}
    all_metrics = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        cv_scores = cross_val_score(model, X_train_bal, y_train_bal, cv=cv, scoring="roc_auc")
        model.fit(X_train_bal, y_train_bal)
        metrics, y_proba, y_pred = evaluate_model(name, model, X_test_scaled, y_test, cv_scores)
        results[name] = (model, metrics, y_proba)
        all_metrics.append(metrics)
        plot_confusion_matrix(name, y_test, y_pred)

    # Save comparison plots
    plot_roc_curves(results, X_test_scaled, y_test)
    plot_precision_recall(results, X_test_scaled, y_test)

    # Pick best model by test ROC-AUC
    best_name = max(all_metrics, key=lambda m: m["roc_auc"])["model"]
    best_model = results[best_name][0]
    print(f"\n>>> Best model: {best_name}")

    # SHAP explainability - use Random Forest specifically (tree-based, SHAP-friendly)
    # even if the ensemble wins on raw AUC, RF gives the clearest clinician-facing explanations
    rf_model = models["Random Forest"]
    X_test_df = pd.DataFrame(X_test_scaled, columns=FEATURE_COLS)
    print("\nGenerating SHAP explainability plots...")
    plot_shap_summary(rf_model, pd.DataFrame(X_train_bal, columns=FEATURE_COLS), X_test_df)

    # Save artifacts for the Streamlit app
    joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
    joblib.dump(rf_model, f"{MODEL_DIR}/rf_model_for_shap.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    with open(f"{MODEL_DIR}/best_model_name.txt", "w") as f:
        f.write(best_name)

    # Save metrics table
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(f"{REPORT_DIR}/model_comparison.csv", index=False)
    print(f"\nSaved model comparison to {REPORT_DIR}/model_comparison.csv")
    print(metrics_df.to_string(index=False))

    print("\nAll artifacts saved. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
