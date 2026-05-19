<h1 align="left">
ECG Arrhythmia Classification Using Ensemble Methods
</h1>

<p align="left">
This is a machine learning project for ECG signal analysis and arrhythmia detection. The project uses the MIT-BIH Arrhythmia Database.
</p>

---

# Project Description

This project creates a system that can classify ECG signals. The system separates normal heartbeats from arrhythmic heartbeats.

Several machine learning methods were used:

- Logistic Regression
- Random Forest
- Soft Voting Ensemble

The SMOTE method was also used to reduce the class imbalance problem.

---

# Project Goal

The goal of this project is to create a classification system that can:

- process ECG signals automatically,
- extract heartbeats from MIT-BIH records,
- classify heartbeats as normal or arrhythmic,
- reduce the class imbalance problem,
- show the results using graphs.

---

# Methods Used

## Logistic Regression

Logistic Regression was used as the basic model.

Main features:

- it trains quickly,
- the results are easy to understand,
- it works well with standardized features.

---

## Random Forest

Random Forest is a method that uses many decision trees.

Main features:

- it can learn nonlinear relationships,
- it is more resistant to noise,
- it works well with complex signal features.

---

## Soft Voting Ensemble

The final model was built using an ensemble method.

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
