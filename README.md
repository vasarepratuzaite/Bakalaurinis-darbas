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

# Thesis Aim and Tasks

The aim of this thesis is to create a model that can classify electrocardiogram signals using signal processing and machine learning methods. The model should identify whether the heart rhythm follows normal patterns.

To achieve this aim, the following tasks are set:

1. Analyse the main types of medical signals and their features.
2. Review biological signal processing methods.
3. Discuss machine learning and deep learning methods used for time series analysis.
4. Present the developed ECG signal classification model.
5. Evaluate the model’s accuracy and its possible use in diagnostics.

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
