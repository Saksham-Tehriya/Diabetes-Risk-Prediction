"""Reproducible diabetes classification experiment with leakage-safe CV."""
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
    classification_report, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "diabetes.csv"
MODEL_DIR = ROOT / "saved_models"
REPORT_DIR = ROOT / "reports"
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
TARGET = "Outcome"
ZERO_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df = df[FEATURES + [TARGET]].copy()
    df[ZERO_MISSING] = df[ZERO_MISSING].replace(0, np.nan)
    return df


def build_models():
    lr = LogisticRegression(max_iter=1500, random_state=SEED)
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3,
                                class_weight="balanced", random_state=SEED, n_jobs=-1)
    ada = AdaBoostClassifier(n_estimators=150, learning_rate=0.5, random_state=SEED)
    stack = StackingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("ada", ada)],
        final_estimator=LogisticRegression(max_iter=1500), cv=5, n_jobs=-1)
    return {"Logistic Regression": lr, "Random Forest": rf,
            "AdaBoost": ada, "Stacking Ensemble": stack}


def make_pipeline(estimator):
    # Imputation, scaling, and SMOTE are refit inside each CV fold.
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()),
                     ("smote", SMOTE(random_state=SEED)),
                     ("model", estimator)])


def plot_curves(results, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, item in results.items():
        fpr, tpr, _ = roc_curve(y_test, item["proba"])
        prec, rec, _ = precision_recall_curve(y_test, item["proba"])
        auc = roc_auc_score(y_test, item["proba"])
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {auc:.2f})")
        axes[1].plot(rec, prec, label=name)
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    axes[0].set(title="ROC curves · held-out test set", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(title="Precision–recall curves · held-out test set", xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "model_curves.png", dpi=180)
    plt.close(fig)


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    df = load_data()
    X, y = df[FEATURES], df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, rows = {}, []

    print(f"Dataset: {len(df)} rows · {len(FEATURES)} features · positive class {y.mean():.1%}")
    print("Evaluation: stratified 80/20 holdout; all preprocessing and SMOTE are fit within CV folds.")
    for name, estimator in build_models().items():
        pipe = make_pipeline(estimator)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        pipe.fit(X_train, y_train)
        pred, proba = pipe.predict(X_test), pipe.predict_proba(X_test)[:, 1]
        row = {"model": name,
               "accuracy": accuracy_score(y_test, pred),
               "balanced_accuracy": balanced_accuracy_score(y_test, pred),
               "precision": precision_score(y_test, pred, zero_division=0),
               "recall": recall_score(y_test, pred, zero_division=0),
               "specificity": recall_score(y_test, pred, pos_label=0, zero_division=0),
               "f1_score": f1_score(y_test, pred, zero_division=0),
               "roc_auc": roc_auc_score(y_test, proba),
               "brier_score": brier_score_loss(y_test, proba),
               "cv_auc_mean": scores.mean(), "cv_auc_std": scores.std()}
        rows.append(row)
        results[name] = {"pipeline": pipe, "pred": pred, "proba": proba}
        print(f"\n{name}: CV ROC-AUC {scores.mean():.3f} ± {scores.std():.3f}; test ROC-AUC {row['roc_auc']:.3f}")
        print(classification_report(y_test, pred, target_names=["No diabetes", "Diabetes"], zero_division=0))
        fig, ax = plt.subplots(figsize=(4.8, 4))
        sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No diabetes", "Diabetes"], yticklabels=["No diabetes", "Diabetes"], ax=ax)
        ax.set(title=f"{name} · held-out test", xlabel="Predicted", ylabel="Actual")
        fig.tight_layout()
        fig.savefig(REPORT_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=170)
        plt.close(fig)

    metrics = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    metrics.to_csv(REPORT_DIR / "model_comparison.csv", index=False, float_format="%.4f")
    plot_curves(results, y_test)
    winner = metrics.sort_values("cv_auc_mean", ascending=False).iloc[0]["model"]
    joblib.dump(results[winner]["pipeline"], MODEL_DIR / "best_model.pkl")
    (MODEL_DIR / "best_model_name.txt").write_text(winner, encoding="utf-8")

    # Tree-based global explanation; the model itself receives imputed/scaled features.
    rf_pipe = results["Random Forest"]["pipeline"]
    transformed_test = rf_pipe.named_steps["scaler"].transform(
        rf_pipe.named_steps["imputer"].transform(X_test))
    rf = rf_pipe.named_steps["model"]
    joblib.dump(rf_pipe.named_steps["imputer"], MODEL_DIR / "imputer.pkl")
    joblib.dump(rf_pipe.named_steps["scaler"], MODEL_DIR / "scaler.pkl")
    joblib.dump(rf, MODEL_DIR / "rf_model_for_shap.pkl")
    explanation = shap.TreeExplainer(rf)(transformed_test)
    values = explanation.values
    if values.ndim == 3:
        values = values[:, :, 1]
    plt.figure()
    shap.summary_plot(values, transformed_test, feature_names=FEATURES, show=False)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close()
    plt.figure()
    shap.summary_plot(values, transformed_test, feature_names=FEATURES, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "shap_feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close()

    metadata = {"seed": SEED, "rows": int(len(df)), "features": FEATURES,
                "positive_rate": float(y.mean()), "test_size": .2,
                "selection_rule": "highest mean stratified training-fold ROC-AUC; test set reserved for final evaluation",
                "selected_model": winner}
    (REPORT_DIR / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nHeld-out test comparison (models ordered by ROC-AUC):")
    print(metrics.round(3).to_string(index=False))
    print(f"\nSelected model: {winner}. Artifacts saved under {MODEL_DIR} and {REPORT_DIR}.")


if __name__ == "__main__":
    main()
