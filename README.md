# Diabetes Risk Studio

An educational machine-learning project that compares four classifiers on the Pima Indians diabetes dataset and presents a sample prediction with a local SHAP explanation. It is designed for a classroom demo, not for diagnosis or clinical decisions.

## Run the project

Use Python 3.10 or newer in a terminal opened at this folder:

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

Training regenerates the model artifacts in `saved_models/` and evaluation outputs in `reports/`. The app loads paths relative to its own file, so it can be started from another working directory too.

## What the pipeline does

1. Loads the included 768-row dataset and treats zero values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI as missing, then imputes medians.
2. Creates a stratified 80/20 train/test split.
3. Compares Logistic Regression, Random Forest, AdaBoost, and a Stacking Ensemble.
4. Runs stratified five-fold ROC-AUC cross-validation. Imputation, scaling, and SMOTE are kept inside an imbalanced-learn pipeline, so each fold learns preprocessing and resampling from its training portion only.
5. Selects a model using mean training-fold ROC-AUC, then reports final metrics on the untouched test split. Reports include ROC-AUC, balanced accuracy, precision, recall, specificity, F1, Brier score, confusion matrices, and ROC / precision-recall curves.
6. Generates a global SHAP summary for the Random Forest and saves the selected prediction pipeline for the app.

## Outputs

- `reports/model_comparison.csv` — held-out test metrics and cross-validation ROC-AUC mean / standard deviation.
- `reports/model_curves.png` — ROC and precision-recall comparisons.
- `reports/confusion_matrix_*.png` — one confusion matrix per model.
- `reports/shap_summary.png` and `reports/shap_feature_importance.png` — global model explanation views.
- `reports/experiment_metadata.json` — seed, dataset size, split, and model-selection rule.
- `saved_models/` — selected pipeline plus the Random Forest explanation artifacts.

## Suggested presentation walkthrough

1. Explain the dataset columns and the meaning of the binary `Outcome` label.
2. Point out the zeros treated as missing and why imputation is fitted inside the pipeline.
3. Explain why SMOTE is applied only to training folds and never to the held-out test data.
4. Compare cross-validation stability (mean ± standard deviation) with the one-time test metrics; discuss precision and recall together.
5. Enter a sample profile in the app, then explain that SHAP describes this model's feature contributions, not causation.
6. Close with the limits below and possible next steps.

## Important limitations

- This small dataset represents Pima Indian women and is not representative of the general population.
- A single held-out split has uncertainty; metrics are educational comparisons, not evidence of clinical performance.
- The displayed low / moderate / high bands are illustrative probability ranges, not validated medical thresholds.
- SHAP describes how this fitted model arrived at a score. It does not establish medical causes.
- This project is not a medical diagnostic device. Any health concern should be discussed with a qualified healthcare professional.

## Next steps

Potential extensions include repeated nested cross-validation, probability calibration and decision-curve analysis, fairness checks across demographic groups, and validation on an independent dataset.
