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

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)

warnings.filterwarnings("ignore")


@dataclass
class Config:
    data_dir: str
    channel: int = 0
    before: int = 90
    after: int = 90
    normal_symbols: Tuple[str, ...] = ("N", "L", "R", "e", "j", "/")
    ignore_symbols: Tuple[str, ...] = ("+", "~", "|", "[", "]", "!", '"', "x")
    test_size: float = 0.25
    random_state: int = 42
    model_output_path: str = "random_forest_model.joblib"
    metadata_output_path: str = "random_forest_metadata.json"


def find_complete_records(data_dir: str) -> List[str]:
    records = []
    for hea_path in glob.glob(os.path.join(data_dir, "*.hea")):
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
        feats = [self._extract_features_one_beat(beat) for beat in X]
        return np.array(feats, dtype=np.float32)

    @staticmethod
    def _safe_zero_crossings(signal: np.ndarray) -> float:
        signs = np.sign(signal)
        return float(np.sum(np.diff(signs) != 0))

    @staticmethod
    def _extract_features_one_beat(beat: np.ndarray):
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
            zero_crossings, width_proxy, peak_index, trough_index
        ]


def zscore_normalize(signal: np.ndarray) -> np.ndarray:
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)


def extract_beats_from_record(record_name: str, config: Config):
    record_path = os.path.join(config.data_dir, record_name)
    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, "atr")

    signal = record.p_signal[:, config.channel]

    X_beats = []
    y_labels = []
    symbols = []
    r_locs = []

    for sample, symbol in zip(ann.sample, ann.symbol):
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

        X_beats.append(beat.astype(np.float32))
        y_labels.append(label)
        symbols.append(symbol)
        r_locs.append(sample)

    return (
        np.array(X_beats, dtype=np.float32),
        np.array(y_labels, dtype=np.int32),
        symbols,
        np.array(r_locs, dtype=np.int32)
    )


def build_dataset(config: Config):
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

            print(
                f"Įrašas {rec}: dūžių={len(y_rec)}, "
                f"normalių={int(np.sum(y_rec == 0))}, "
                f"aritminių={int(np.sum(y_rec == 1))}"
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

    if not X_all:
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


def split_train_test_by_record(X, y, groups, config: Config):
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


def build_random_forest_model(config: Config):
    return Pipeline(steps=[
        ("feature_extractor", ECGFeatureExtractor()),
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=config.random_state
        ))
    ])


def evaluate_binary_classifier(model, X_test, y_test, title="Modelis"):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob))
    }

    print(f"\n{'=' * 60}")
    print(f"VERTINIMAS: {title}")
    print(f"{'=' * 60}")
    print("Balanced accuracy:", round(metrics["balanced_accuracy"], 4))
    print("F1 score:         ", round(metrics["f1"], 4))
    print("Precision:        ", round(metrics["precision"], 4))
    print("Recall:           ", round(metrics["recall"], 4))
    print("ROC AUC:          ", round(metrics["roc_auc"], 4))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    return metrics


def save_model_and_metadata(model, metrics: Dict, training_info: Dict, config: Config):
    joblib.dump(model, config.model_output_path)

    metadata = {
        "model_name": "Random Forest",
        "config": asdict(config),
        "metrics": metrics,
        "training_info": training_info
    }

    with open(config.metadata_output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nModelis išsaugotas į:", config.model_output_path)
    print("Metadata išsaugota į:", config.metadata_output_path)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "dataset")

    config = Config(
        data_dir=data_dir,
        model_output_path=os.path.join(script_dir, "random_forest_model.joblib"),
        metadata_output_path=os.path.join(script_dir, "random_forest_metadata.json")
    )

    print("\nPradedamas Random Forest modelio mokymas...\n")

    X, y, groups, _ = build_dataset(config)
    X_train, X_test, y_train, y_test, groups_train, groups_test = split_train_test_by_record(X, y, groups, config)

    print("\nTrain įrašai:", sorted(np.unique(groups_train).tolist()))
    print("Test įrašai: ", sorted(np.unique(groups_test).tolist()))

    model = build_random_forest_model(config)
    model.fit(X_train, y_train)

    metrics = evaluate_binary_classifier(model, X_test, y_test, title="Random Forest")

    training_info = {
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "train_records": sorted(np.unique(groups_train).tolist()),
        "test_records": sorted(np.unique(groups_test).tolist()),
        "dataset_size": int(len(y)),
        "normal_count": int(np.sum(y == 0)),
        "abnormal_count": int(np.sum(y == 1))
    }

    save_model_and_metadata(model, metrics, training_info, config)
    print("\nBaigta.")


if __name__ == "__main__":
    main()