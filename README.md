# Diabetes Risk Prediction — ML Project

A college-level machine learning project that predicts diabetes risk from routine
clinical measurements, inspired by the "Diabetes Prediction: Transforming Healthcare
with Machine Learning" brief. It covers the full pipeline the brief describes:
data preprocessing, handling class imbalance, training multiple models, rigorous
evaluation, explainability with SHAP, and a simple deployed web app.

## What's inside

```
diabetes_ml_project/
├── data/
│   └── diabetes.csv                  # Pima Indians Diabetes dataset (768 patients)
├── notebooks/
│   └── Diabetes_Prediction_EDA_and_Modeling.ipynb   # Full EDA + modeling walkthrough
├── train_models.py                   # End-to-end training script (run this first)
├── app.py                            # Streamlit "Visualized Checkup" web app
├── saved_models/                     # Created by train_models.py (model + scaler)
├── reports/                          # Created by train_models.py (plots, metrics CSV)
└── requirements.txt
```

## The dataset

The **Pima Indians Diabetes Dataset**: 768 patients, 8 features
(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
DiabetesPedigreeFunction, Age) and a binary `Outcome` label. It's the standard,
freely available dataset used for this kind of tabular medical ML project — a
realistic scope for a course project (a hospital-scale system would need much
larger, messier, and access-controlled data).

## How to run it

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train all models, generate evaluation plots and SHAP explanations
python train_models.py

# 3. Launch the interactive app
streamlit run app.py
```

Or open `notebooks/Diabetes_Prediction_EDA_and_Modeling.ipynb` in Jupyter to see the
full walkthrough (EDA → preprocessing → SMOTE → model training → evaluation → SHAP)
with all outputs already rendered — good for showing your professor the reasoning
step by step.

## Pipeline (matches the project brief)

1. **Data Collection** — load the dataset.
2. **Preprocessing** — several columns use `0` to mean "not recorded"
   (Glucose, BloodPressure, SkinThickness, Insulin, BMI cannot really be 0 in a
   living patient). These are treated as missing and imputed with the median.
   Features are then standardized with `StandardScaler`.
3. **Handling imbalance** — only ~35% of patients are diabetic. We apply **SMOTE**
   oversampling to the *training set only* (after the train/test split, to avoid
   data leakage) so models don't just learn to predict "no diabetes" every time.
4. **Model training** — four models are trained and compared:
   - Logistic Regression (interpretable baseline)
   - Random Forest (robust to mixed feature types/non-linearities)
   - AdaBoost (boosting)
   - Stacking Ensemble (Logistic Regression + Random Forest + AdaBoost, combined
     via a logistic-regression meta-learner)
5. **Evaluation** — stratified 5-fold cross-validation on ROC-AUC, plus test-set
   accuracy, precision, recall, F1, ROC-AUC, ROC curves, precision-recall curves,
   and confusion matrices (all saved to `reports/`).
6. **Explainability** — SHAP summary and feature-importance plots show *which*
   features drive predictions (Glucose, BMI, and Age dominate — consistent with
   clinical knowledge, which is a good sanity check on the model).
7. **Deployment** — `app.py` is a small Streamlit app: enter a patient's values,
   get a risk-probability "gauge", a SHAP waterfall chart explaining that specific
   prediction, and a downloadable PDF summary report.

## Results (example run)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.71 | 0.58 | 0.67 | 0.62 | 0.81 | 0.84 |
| Random Forest | 0.73 | 0.58 | 0.78 | 0.67 | 0.82 | 0.88 |
| AdaBoost | 0.75 | 0.61 | 0.80 | 0.69 | 0.83 | 0.85 |
| Stacking Ensemble | 0.75 | 0.61 | 0.78 | 0.68 | 0.82 | 0.88 |

(Exact numbers can vary slightly by random seed/environment — re-run
`train_models.py` to regenerate `reports/model_comparison.csv`.)

Recall and ROC-AUC matter more than raw accuracy here: in a screening context,
missing an at-risk patient (false negative) is costlier than a false alarm.

## Limitations & honest caveats (good to mention to your professor)

- The dataset is small (768 rows) and from a single population (Pima Indian
  women) — it will **not** generalize to a general population without
  re-validation.
- This is a **demo/educational project**, not a validated medical device. The
  app includes a disclaimer for this reason.
- Zero-imputation with the median is a simple strategy; more advanced imputation
  (KNN, iterative imputer) could be explored as a next step.

## Possible extensions

- Hyperparameter tuning (GridSearchCV / Optuna)
- Try gradient boosting libraries (XGBoost, LightGBM, CatBoost)
- External validation on a second dataset
- User authentication and persistent patient history in the app
