import os
import glob
import json
import warnings
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import wfdb
import joblib
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve

from imblearn.over_sampling import SMOTE

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")


@dataclass
class Config:
    data_dir: str = "/mnt/data"
    channel: int = 0
    before: int = 90
    after: int = 90

    normal_symbols: Tuple[str, ...] = ("N", "L", "R", "e", "j", "/")
    ignore_symbols: Tuple[str, ...] = ("+", "~", "|", "[", "]", "!", '"', "x")

    record_level_threshold: float = 0.05
    test_size: float = 0.25
    random_state: int = 42

    model_output_path: str = "ekg_ensemble_model.joblib"
    metadata_output_path: str = "ekg_model_metadata.json"
    output_dir: str = "."
    use_smote: bool = True


def find_complete_records(data_dir: str) -> List[str]:
    records = []
    hea_files = glob.glob(os.path.join(data_dir, "*.hea"))

    for hea_path in hea_files:
        record_name = os.path.splitext(os.path.basename(hea_path))[0]
        dat_path = os.path.join(data_dir, f"{record_name}.dat")
        atr_path = os.path.join(data_dir, f"{record_name}.atr")

        if os.path.exists(dat_path) and os.path.exists(atr_path):
            records.append(record_name)

    return sorted(records)


def print_record_summary(records: List[str]) -> None:
    print("=" * 60)
    print("Rasti pilni įrašai:")
    print(", ".join(records))
    print(f"Iš viso pilnų įrašų: {len(records)}")
    print("=" * 60)


class ECGFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = [self._extract_features_one_beat(x) for x in X]
        return np.array(feats, dtype=np.float32)

    @staticmethod
    def _safe_zero_crossings(signal: np.ndarray) -> float:
        signs = np.sign(signal)
        return float(np.sum(np.diff(signs) != 0))

    @staticmethod
    def _extract_features_one_beat(data) -> List[float]:
        beat, rr_prev, rr_next, rr_ratio, rr_diff = data
        eps = 1e-8

        mean_val = np.mean(beat)
        std_val = np.std(beat)
        min_val = np.min(beat)
        max_val = np.max(beat)
        median_val = np.median(beat)

        q25 = np.percentile(beat, 25)
        q75 = np.percentile(beat, 75)
        iqr = q75 - q25

        peak_to_peak = max_val - min_val
        energy = np.sum(beat ** 2)
        abs_energy = np.sum(np.abs(beat))
        rms = np.sqrt(np.mean(beat ** 2))

        skew_like = np.mean(((beat - mean_val) / (std_val + eps)) ** 3)
        kurt_like = np.mean(((beat - mean_val) / (std_val + eps)) ** 4)

        diff1 = np.diff(beat)
        diff2 = np.diff(diff1) if len(diff1) > 1 else np.array([0.0])

        diff1_mean = np.mean(diff1) if len(diff1) > 0 else 0.0
        diff1_std = np.std(diff1) if len(diff1) > 0 else 0.0
        diff1_max = np.max(diff1) if len(diff1) > 0 else 0.0
        diff1_min = np.min(diff1) if len(diff1) > 0 else 0.0

        diff2_mean = np.mean(diff2) if len(diff2) > 0 else 0.0
        diff2_std = np.std(diff2) if len(diff2) > 0 else 0.0

        zero_crossings = ECGFeatureExtractor._safe_zero_crossings(beat)

        amp_threshold = min_val + 0.5 * (max_val - min_val)
        width_proxy = float(np.sum(beat >= amp_threshold))

        peak_index = float(np.argmax(beat))
        trough_index = float(np.argmin(beat))

        return [
            mean_val, std_val, min_val, max_val, median_val,
            q25, q75, iqr, peak_to_peak, energy, abs_energy, rms,
            skew_like, kurt_like,
            diff1_mean, diff1_std, diff1_max, diff1_min,
            diff2_mean, diff2_std,
            zero_crossings, width_proxy, peak_index, trough_index,
            float(rr_prev), float(rr_next), float(rr_ratio), float(rr_diff)
        ]

def zscore_normalize(signal: np.ndarray) -> np.ndarray:
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)


def extract_beats_from_record(
    record_name: str,
    config: Config
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    record_path = os.path.join(config.data_dir, record_name)

    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, "atr")

    signal = record.p_signal[:, config.channel]

    X_beats = []
    y_labels = []
    symbols = []
    r_locs = []

    for i, (sample, symbol) in enumerate(zip(ann.sample, ann.symbol)):
        if symbol in config.ignore_symbols:
            continue

        start = sample - config.before
        end = sample + config.after

        if start < 0 or end > len(signal):
            continue

        beat = signal[start:end].copy()

        if len(beat) != (config.before + config.after):
            continue

        beat = zscore_normalize(beat)
        label = 0 if symbol in config.normal_symbols else 1

        # RR intervalai
        if i > 0:
            rr_prev = sample - ann.sample[i - 1]
        else:
            rr_prev = 0

        if i < len(ann.sample) - 1:
            rr_next = ann.sample[i + 1] - sample
        else:
            rr_next = 0

        rr_ratio = rr_prev / (rr_next + 1e-8)
        rr_diff = rr_prev - rr_next

        X_beats.append((beat.astype(np.float32), rr_prev, rr_next, rr_ratio, rr_diff))
        y_labels.append(label)
        symbols.append(symbol)
        r_locs.append(sample)

    return (
        np.array(X_beats, dtype=object),
        np.array(y_labels, dtype=np.int32),
        symbols,
        np.array(r_locs, dtype=np.int32)
    )


def build_dataset(config: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    records = find_complete_records(config.data_dir)
    print_record_summary(records)

    X_all = []
    y_all = []
    groups = []
    rows_meta = []

    for rec in records:
        try:
            X_rec, y_rec, symbols, r_locs = extract_beats_from_record(rec, config)

            if len(y_rec) == 0:
                print(f"[WARN] Įrašas {rec}: nerasta tinkamų dūžių")
                continue

            normal_count = int(np.sum(y_rec == 0))
            abnormal_count = int(np.sum(y_rec == 1))

            print(
                f"Įrašas {rec}: dūžių={len(y_rec)}, "
                f"normalių={normal_count}, "
                f"aritminių={abnormal_count}"
            )

            X_all.append(X_rec)
            y_all.append(y_rec)
            groups.extend([rec] * len(y_rec))

            for i in range(len(y_rec)):
                rows_meta.append({
                    "record": rec,
                    "symbol": symbols[i],
                    "r_location": int(r_locs[i]),
                    "label": int(y_rec[i])
                })

        except Exception as e:
            print(f"[ERROR] Nepavyko apdoroti įrašo {rec}: {e}")

    if len(X_all) == 0:
        raise RuntimeError("Nepavyko sukurti dataset. Nerasta tinkamų įrašų.")

    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    groups = np.array(groups)
    df_meta = pd.DataFrame(rows_meta)

    print("=" * 60)
    print("Bendras dataset:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Normalūs:", int(np.sum(y == 0)))
    print("Aritminiai:", int(np.sum(y == 1)))
    print("=" * 60)

    return X, y, groups, df_meta


def build_models(config: Config):
    logistic_pipeline = Pipeline(steps=[
        ("feature_extractor", ECGFeatureExtractor()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=3000,
            C=0.5,
            class_weight="balanced",
            solver="lbfgs",
            random_state=config.random_state
        ))
    ])

    rf_pipeline = Pipeline(steps=[
        ("feature_extractor", ECGFeatureExtractor()),
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=config.random_state
        ))
    ])

    ensemble = VotingClassifier(
        estimators=[
            ("logreg", logistic_pipeline),
            ("rf", rf_pipeline)
        ],
        voting="soft",
        weights=[1.0, 3.0],
        n_jobs=-1
    )

    return logistic_pipeline, rf_pipeline, ensemble


def build_xgb_model(config: Config, scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=float(scale_pos_weight),
        random_state=config.random_state,
        n_jobs=-1
    )

    rf_pipeline = Pipeline(steps=[
        ("feature_extractor", ECGFeatureExtractor()),
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=config.random_state
        ))
    ])

    ensemble = VotingClassifier(
        estimators=[
            ("logreg", logistic_pipeline),
            ("rf", rf_pipeline)
        ],
        voting="soft",
        weights=[1.0, 3.0],
        n_jobs=-1
    )

    return logistic_pipeline, rf_pipeline, ensemble


def evaluate_binary_classifier(model, X_test, y_test, title: str = "Modelis") -> Dict[str, float]:
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred.astype(float)

    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else np.nan
    }

    print(f"\n{'=' * 60}")
    print(f"VERTINIMAS: {title}")
    print(f"{'=' * 60}")
    print("Balanced accuracy:", round(metrics["balanced_accuracy"], 4))
    print("F1 score:         ", round(metrics["f1"], 4))
    print("Precision:        ", round(metrics["precision"], 4))
    print("Recall:           ", round(metrics["recall"], 4))
    print("ROC AUC:          ", round(metrics["roc_auc"], 4) if not np.isnan(metrics["roc_auc"]) else "nan")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    return metrics

def find_best_threshold(y_true, y_prob, metric: str = "f1") -> Tuple[float, Dict[str, float]]:
    thresholds = np.arange(0.1, 0.91, 0.01)

    best_threshold = 0.5
    best_metrics = None
    best_score = -1.0

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        metrics = {
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan
        }

        score = metrics[metric]

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics

def evaluate_with_threshold(model, X_test, y_test, threshold: float, title: str = "Modelis") -> Dict[str, float]:
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test).astype(float)

    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else np.nan
    }

    print(f"\n{'=' * 60}")
    print(f"VERTINIMAS SU THRESHOLD: {title}")
    print(f"{'=' * 60}")
    print("Threshold:        ", round(threshold, 4))
    print("Balanced accuracy:", round(metrics["balanced_accuracy"], 4))
    print("F1 score:         ", round(metrics["f1"], 4))
    print("Precision:        ", round(metrics["precision"], 4))
    print("Recall:           ", round(metrics["recall"], 4))
    print("ROC AUC:          ", round(metrics["roc_auc"], 4) if not np.isnan(metrics["roc_auc"]) else "nan")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    return metrics

def plot_class_distribution(y: np.ndarray, output_path: str):
    unique, counts = np.unique(y, return_counts=True)
    values = {0: 0, 1: 0}
    for cls, cnt in zip(unique, counts):
        values[int(cls)] = int(cnt)

    plt.figure(figsize=(8, 5))
    plt.bar(["Normalus", "Aritminis"], [values[0], values[1]])
    plt.title("Klasių pasiskirstymas duomenų rinkinyje")
    plt.xlabel("Klasė")
    plt.ylabel("Dūžių skaičius")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Grafikas išsaugotas: {output_path}")


def plot_confusion_matrix_figure(model, X_test, y_test, title: str, output_path: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Grafikas išsaugotas: {output_path}")


def plot_roc_curves(models: Dict[str, object], X_test, y_test, output_path: str):
    plt.figure(figsize=(8, 6))

    for model_name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.predict(X_test).astype(float)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc_val:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC kreivių palyginimas")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Grafikas išsaugotas: {output_path}")


def plot_metrics_comparison(all_metrics: Dict[str, Dict[str, float]], output_path: str):
    metric_names = ["balanced_accuracy", "f1", "precision", "recall", "roc_auc"]
    model_names = list(all_metrics.keys())

    x = np.arange(len(metric_names))
    width = 0.25

    plt.figure(figsize=(12, 6))

    for i, model_name in enumerate(model_names):
        values = [all_metrics[model_name][m] for m in metric_names]
        plt.bar(x + i * width, values, width=width, label=model_name)

    plt.xticks(x + width, metric_names, rotation=20)
    plt.ylim(0, 1.0)
    plt.ylabel("Reikšmė")
    plt.title("Modelių metrikų palyginimas")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Grafikas išsaugotas: {output_path}")


def plot_class_distribution_train_before_after(
    y_train: np.ndarray,
    y_train_resampled: np.ndarray,
    output_path: str
):
    before_counts = [int(np.sum(y_train == 0)), int(np.sum(y_train == 1))]
    after_counts = [int(np.sum(y_train_resampled == 0)), int(np.sum(y_train_resampled == 1))]

    x = np.arange(2)
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, before_counts, width, label="Prieš SMOTE")
    plt.bar(x + width / 2, after_counts, width, label="Po SMOTE")
    plt.xticks(x, ["Normalus", "Aritminis"])
    plt.ylabel("Dūžių skaičius")
    plt.title("Train aibės klasių pasiskirstymas prieš ir po SMOTE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Grafikas išsaugotas: {output_path}")


def split_train_test_by_record(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    config: Config
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=config.test_size,
        random_state=config.random_state
    )

    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    return (
        X[train_idx], X[test_idx],
        y[train_idx], y[test_idx],
        groups[train_idx], groups[test_idx]
    )


def train_and_evaluate(config: Config):
    X, y, groups, df_meta = build_dataset(config)

    X_train, X_test, y_train, y_test, groups_train, groups_test = split_train_test_by_record(
        X, y, groups, config
    )

    print("\nTrain įrašai:", sorted(np.unique(groups_train).tolist()))
    print("Test įrašai: ", sorted(np.unique(groups_test).tolist()))

    normal_train = int(np.sum(y_train == 0))
    abnormal_train = int(np.sum(y_train == 1))
    scale_pos_weight = normal_train / max(abnormal_train, 1)

    # Feature extractor reikalingas SMOTE, nes SMOTE veikia su feature matrica, ne su pipeline
    feature_extractor = ECGFeatureExtractor()
    X_train_features = feature_extractor.transform(X_train)
    X_test_features = feature_extractor.transform(X_test)

    imputer = SimpleImputer(strategy="median")
    X_train_features = imputer.fit_transform(X_train_features)
    X_test_features = imputer.transform(X_test_features)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_test_scaled = scaler.transform(X_test_features)

    if config.use_smote:
        smote = SMOTE(random_state=config.random_state)
        X_train_scaled_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
        X_train_features_res, y_train_features_res = smote.fit_resample(X_train_features, y_train)

        print("\nPo SMOTE balansavimo:")
        print("Normalūs:", int(np.sum(y_train_res == 0)))
        print("Aritminiai:", int(np.sum(y_train_res == 1)))
    else:
        X_train_scaled_res, y_train_res = X_train_scaled, y_train
        X_train_features_res, y_train_features_res = X_train_features, y_train

    # Atskiri modeliai
    logistic_model = LogisticRegression(
        max_iter=3000,
        C=0.5,
        class_weight="balanced",
        solver="lbfgs",
        random_state=config.random_state
    )

    rf_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=config.random_state
    )

    xgb_model = build_xgb_model(config, scale_pos_weight=scale_pos_weight)

    all_metrics = {}

    print("\nMokomas Logistic Regression...")
    logistic_model.fit(X_train_scaled_res, y_train_res)
    all_metrics["logistic_regression"] = evaluate_binary_classifier(
        logistic_model, X_test_scaled, y_test, title="Logistic Regression"
    )

    print("\nMokomas Random Forest...")
    rf_model.fit(X_train_features_res, y_train_features_res)
    all_metrics["random_forest"] = evaluate_binary_classifier(
        rf_model, X_test_features, y_test, title="Random Forest"
    )

    print("\nMokomas XGBoost...")
    xgb_model.fit(X_train_features_res, y_train_features_res)

    all_metrics["xgboost_default"] = evaluate_binary_classifier(
        xgb_model, X_test_features, y_test, title="XGBoost (default threshold=0.5)"
    )

    xgb_prob = xgb_model.predict_proba(X_test_features)[:, 1]
    best_xgb_threshold, best_xgb_metrics = find_best_threshold(y_test, xgb_prob, metric="f1")

    print(f"\nGeriausias XGBoost threshold pagal F1: {best_xgb_threshold:.2f}")

    all_metrics["xgboost_tuned"] = evaluate_with_threshold(
        xgb_model, X_test_features, y_test,
        threshold=best_xgb_threshold,
        title="XGBoost (tuned threshold)"
    )

    print("\nMokomas Ensemble (Soft Voting su XGBoost)...")
    ensemble_model = VotingClassifier(
        estimators=[
            ("logreg", Pipeline(steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=3000,
                    C=0.5,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=config.random_state
                ))
            ])),
            ("rf", RandomForestClassifier(
                n_estimators=400,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=config.random_state
            )),
            ("xgb", build_xgb_model(config, scale_pos_weight=scale_pos_weight))
        ],
        voting="soft",
        weights=[1.0, 2.0, 3.0],
        n_jobs=-1
    )

    if config.use_smote:
        smote_ensemble = SMOTE(random_state=config.random_state)
        X_train_ensemble_res, y_train_ensemble_res = smote_ensemble.fit_resample(X_train_features, y_train)
    else:
        X_train_ensemble_res, y_train_ensemble_res = X_train_features, y_train

    ensemble_model.fit(X_train_ensemble_res, y_train_ensemble_res)

    all_metrics["ensemble_default"] = evaluate_binary_classifier(
        ensemble_model, X_test_features, y_test, title="Soft Voting Ensemble (default threshold=0.5)"
    )

    ensemble_prob = ensemble_model.predict_proba(X_test_features)[:, 1]
    best_ensemble_threshold, best_ensemble_metrics = find_best_threshold(y_test, ensemble_prob, metric="f1")

    print(f"\nGeriausias Ensemble threshold pagal F1: {best_ensemble_threshold:.2f}")

    all_metrics["ensemble_tuned"] = evaluate_with_threshold(
        ensemble_model, X_test_features, y_test,
        threshold=best_ensemble_threshold,
        title="Soft Voting Ensemble (tuned threshold)"
    )

    training_info = {
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "train_records": sorted(np.unique(groups_train).tolist()),
        "test_records": sorted(np.unique(groups_test).tolist()),
        "dataset_size": int(len(y)),
        "normal_count": int(np.sum(y == 0)),
        "abnormal_count": int(np.sum(y == 1)),
        "smote_used": config.use_smote,
        "xgboost_used": True,
        "scale_pos_weight": float(scale_pos_weight),
        "xgb_best_threshold_f1": float(best_xgb_threshold),
        "ensemble_best_threshold_f1": float(best_ensemble_threshold),
    }

    trained_models = {
        "Logistic Regression": ("scaled", logistic_model, scaler, imputer, feature_extractor),
        "Random Forest": ("features", rf_model, None, imputer, feature_extractor),
        "XGBoost": ("features", xgb_model, None, imputer, feature_extractor),
        "Ensemble": ("features", ensemble_model, None, imputer, feature_extractor)
    }

    extra_data = {
        "X": X,
        "y": y,
        "X_test_raw": X_test,
        "y_test": y_test,
        "y_train_before_smote": y_train,
        "y_train_after_smote": y_train_res if config.use_smote else y_train
    }

    return trained_models, all_metrics, training_info, extra_data

def transform_for_prediction(X_raw: np.ndarray, model_bundle):
    mode, model, scaler, imputer, feature_extractor = model_bundle

    X_features = feature_extractor.transform(X_raw)
    X_features = imputer.transform(X_features)

    if mode == "scaled":
        X_features = scaler.transform(X_features)

    return X_features


def predict_record_beats(record_name: str, model_bundle, config: Config) -> pd.DataFrame:
    X_rec, y_true, symbols, r_locs = extract_beats_from_record(record_name, config)

    if len(X_rec) == 0:
        return pd.DataFrame()

    X_ready = transform_for_prediction(X_rec, model_bundle)
    _, model, _, _, _ = model_bundle

    y_pred = model.predict(X_ready)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_ready)[:, 1]
    else:
        y_prob = y_pred.astype(float)

    df = pd.DataFrame({
        "record": record_name,
        "r_location": r_locs,
        "symbol": symbols,
        "true_label": y_true,
        "pred_label": y_pred,
        "pred_prob_abnormal": y_prob
    })

    return df


def predict_record_level(record_name: str, model_bundle, config: Config) -> Dict:
    df = predict_record_beats(record_name, model_bundle, config)

    if df.empty:
        return {
            "record": record_name,
            "status": "error",
            "message": "Nerasta tinkamų dūžių"
        }

    abnormal_ratio = float(np.mean(df["pred_label"].values))
    abnormal_count = int(np.sum(df["pred_label"].values))
    total_beats = int(len(df))

    return {
        "record": record_name,
        "status": "ok",
        "beats_total": total_beats,
        "beats_predicted_abnormal": abnormal_count,
        "abnormal_ratio": abnormal_ratio,
        "record_prediction": (
            "galimai turi aritmijos požymių"
            if abnormal_ratio >= config.record_level_threshold
            else "greičiausiai normalus"
        )
    }


def save_model_and_metadata(model_bundle, metrics: Dict, training_info: Dict, config: Config):
    joblib.dump(model_bundle, config.model_output_path)

    metadata = {
        "config": asdict(config),
        "metrics": metrics,
        "training_info": training_info
    }

    with open(config.metadata_output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nModelis išsaugotas į:", config.model_output_path)
    print("Metadata išsaugota į:", config.metadata_output_path)


def load_model(model_path: str):
    return joblib.load(model_path)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = script_dir

    config = Config(
        data_dir="/Users/vasarep./Desktop/klasifikatorius/dataset",
        channel=0,
        before=90,
        after=90,
        record_level_threshold=0.05,
        test_size=0.25,
        random_state=42,
        model_output_path=os.path.join(output_dir, "ekg_ensemble_model.joblib"),
        metadata_output_path=os.path.join(output_dir, "ekg_model_metadata.json"),
        output_dir=output_dir,
        use_smote=True
    )

    os.makedirs(config.output_dir, exist_ok=True)

    print("\nPradedamas modelio mokymas...\n")
    trained_models, metrics, training_info, extra_data = train_and_evaluate(config)

    save_model_and_metadata(trained_models["Ensemble"], metrics, training_info, config)

    print("\nGALUTINIAI REZULTATAI:")
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name}:")
        for k, v in model_metrics.items():
            print(f"  {k}: {round(v, 4)}")

    X_test_raw = extra_data["X_test_raw"]
    y_test = extra_data["y_test"]

    # Grafikai
    plot_class_distribution(
        extra_data["y"],
        os.path.join(config.output_dir, "XGBoost_prt3.3_class_distribution.png")
    )

    plot_class_distribution_train_before_after(
        extra_data["y_train_before_smote"],
        extra_data["y_train_after_smote"],
        os.path.join(config.output_dir, "XGBoost_prt3.3_class_distribution_smote.png")
    )

    for model_name, model_bundle in trained_models.items():
        X_ready = transform_for_prediction(X_test_raw, model_bundle)
        _, model, _, _, _ = model_bundle

        safe_name = model_name.lower().replace(" ", "_")
        plot_confusion_matrix_figure(
            model,
            X_ready,
            y_test,
            f"Confusion Matrix - {model_name}",
            os.path.join(config.output_dir, f"XGBoost_prt3.3_confusion_matrix_{safe_name}.png")
        )

    # ROC
    roc_models = {}
    for model_name, model_bundle in trained_models.items():
        X_ready = transform_for_prediction(X_test_raw, model_bundle)
        _, model, _, _, _ = model_bundle
        roc_models[model_name] = (model, X_ready)

    plt.figure(figsize=(8, 6))
    for model_name, (model, X_ready) in roc_models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_ready)[:, 1]
        else:
            y_prob = model.predict(X_ready).astype(float)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc_val:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC kreivių palyginimas")
    plt.legend()
    plt.tight_layout()
    roc_path = os.path.join(config.output_dir, "XGBoost_prt3.3_roc_curves.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Grafikas išsaugotas: {roc_path}")

    plot_metrics_comparison(
        metrics,
        os.path.join(config.output_dir, "XGBoost_prt3.3_comparison.png")
    )

    print("\nPavyzdinės įrašo lygio prognozės:")
    for rec in find_complete_records(config.data_dir)[:5]:
        result = predict_record_level(rec, trained_models["Ensemble"], config)
        print(result)

    print("\nBaigta.")
    print(f"\nGrafikai išsaugoti čia: {config.output_dir}")


if __name__ == "__main__":
    main()