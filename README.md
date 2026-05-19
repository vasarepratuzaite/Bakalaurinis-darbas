<h1 
ECG Arrhythmia Classification Using Ensemble Methods
</h1>

<p align="center">
A machine learning project for ECG signal analysis and arrhythmia detection using the MIT-BIH Arrhythmia Database.
</p>

---

# Project Description

This project presents an ECG signal classification system designed to automatically distinguish between normal and arrhythmic heartbeats.

Several classification methods and their combination into an ensemble model were used for model development:

- Logistic Regression
- Random Forest
- Soft Voting Ensemble

Additionally, the SMOTE method was applied to reduce the class imbalance problem.

---

# Project Goal

To create a classification system that:

- automatically processes ECG signals,
- extracts heartbeats from MIT-BIH records,
- classifies normal and arrhythmic signals,
- reduces the class imbalance problem,
- presents result analysis in the form of graphs.

---

# Methods Used

## Logistic Regression

Logistic Regression was used as the baseline classifier.

Main characteristics:

- fast training,
- interpretable results,
- stable performance with standardized features.

---

## Random Forest

The Random Forest method consists of multiple decision trees.

Main characteristics:

- ability to model nonlinear relationships,
- robustness to noise,
- good performance with complex signal features.

---

## Soft Voting Ensemble

The final model was created using an ensemble method.

Ensemble structure:

```python
VotingClassifier(
    estimators=[
        ("logreg", logistic_model),
        ("rf", rf_model)
    ],
    voting="soft",
    weights=[1.0, 3.0]
)
