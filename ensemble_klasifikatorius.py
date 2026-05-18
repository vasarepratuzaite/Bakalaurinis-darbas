# ============================================================
# EKG aritmijos klasifikavimas su dviem klasifikatoriais
# ir jų sujungimu į vieną 
#
# Naudojami modeliai:
# 1) Logistic Regression
# 2) Random Forest
# 3) Soft Voting Ensemble
#
# Tinka MIT-BIH formato failams:
# .dat + .hea + .atr
# ============================================================

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


# ============================================================
# 1. KONFIGŪRACIJA
# ============================================================

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


# ============================================================
# 2. PAGALBINĖS FUNKCIJOS FAILAMS
# ============================================================

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


# ============================================================
# 3. POŽYMIŲ IŠTRAUKIMO KLASĖ
# ============================================================

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
    def _extract_features_one_beat(beat: np.ndarray) -> List[float]:
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


# ============================================================
# 4. DUOMENŲ ĮKĖLIMAS IR DŪŽIŲ IŠTRAUKIMAS
# ============================================================

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


# ============================================================
# 5. MODELIO KŪRIMAS
# ============================================================

def build_models(config: Config):
    logistic_pipeline = Pipeline(steps=[
        ("feature_extractor", ECGFeatureExtractor()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=config.random_state
        ))
    ])

    rf_pipeline = Pipeline(steps=[
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

    ensemble = VotingClassifier(
        estimators=[
            ("logreg", logistic_pipeline),
            ("rf", rf_pipeline)
        ],
        voting="soft",
        weights=[1.0, 2.0],
        n_jobs=-1
    )

    return logistic_pipeline, rf_pipeline, ensemble


# ============================================================
# 6. VERTINIMO FUNKCIJOS
# ============================================================

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


# ============================================================
# 6.1 GRAFIKŲ FUNKCIJOS
# ============================================================

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


# ============================================================
# 7. MOKYMAS
# ============================================================

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

    logistic_model, rf_model, ensemble_model = build_models(config)

    all_metrics = {}

    print("\nMokomas Logistic Regression...")
    logistic_model.fit(X_train, y_train)
    all_metrics["logistic_regression"] = evaluate_binary_classifier(
        logistic_model, X_test, y_test, title="Logistic Regression"
    )

    print("\nMokomas Random Forest...")
    rf_model.fit(X_train, y_train)
    all_metrics["random_forest"] = evaluate_binary_classifier(
        rf_model, X_test, y_test, title="Random Forest"
    )

    print("\nMokomas Ensemble (Soft Voting)...")
    ensemble_model.fit(X_train, y_train)
    all_metrics["ensemble"] = evaluate_binary_classifier(
        ensemble_model, X_test, y_test, title="Soft Voting Ensemble"
    )

    training_info = {
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "train_records": sorted(np.unique(groups_train).tolist()),
        "test_records": sorted(np.unique(groups_test).tolist()),
        "dataset_size": int(len(y)),
        "normal_count": int(np.sum(y == 0)),
        "abnormal_count": int(np.sum(y == 1))
    }

    trained_models = {
        "Logistic Regression": logistic_model,
        "Random Forest": rf_model,
        "Ensemble": ensemble_model
    }

    return trained_models, all_metrics, training_info, X, y, X_test, y_test


# ============================================================
# 8. PROGNOZĖ VIENAM ĮRAŠUI
# ============================================================

def predict_record_beats(record_name: str, model, config: Config) -> pd.DataFrame:
    X_rec, y_true, symbols, r_locs = extract_beats_from_record(record_name, config)

    if len(X_rec) == 0:
        return pd.DataFrame()

    y_pred = model.predict(X_rec)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_rec)[:, 1]
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


def predict_record_level(record_name: str, model, config: Config) -> Dict:
    df = predict_record_beats(record_name, model, config)

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


# ============================================================
# 9. IŠSAUGOJIMAS
# ============================================================

def save_model_and_metadata(model, metrics: Dict, training_info: Dict, config: Config):
    joblib.dump(model, config.model_output_path)

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


# ============================================================
# 10. MAIN
# ============================================================

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
        output_dir=output_dir
    )

    os.makedirs(config.output_dir, exist_ok=True)

    print("\nPradedamas modelio mokymas...\n")
    trained_models, metrics, training_info, X, y, X_test, y_test = train_and_evaluate(config)

    save_model_and_metadata(trained_models["Ensemble"], metrics, training_info, config)

    print("\nGALUTINIAI REZULTATAI:")
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name}:")
        for k, v in model_metrics.items():
            print(f"  {k}: {round(v, 4)}")

    plot_class_distribution(
        y,
        os.path.join(config.output_dir, "class_distribution.png")
    )

    plot_confusion_matrix_figure(
        trained_models["Logistic Regression"],
        X_test, y_test,
        "Confusion Matrix - Logistic Regression",
        os.path.join(config.output_dir, "confusion_matrix_logistic.png")
    )

    plot_confusion_matrix_figure(
        trained_models["Random Forest"],
        X_test, y_test,
        "Confusion Matrix - Random Forest",
        os.path.join(config.output_dir, "confusion_matrix_rf.png")
    )

    plot_confusion_matrix_figure(
        trained_models["Ensemble"],
        X_test, y_test,
        "Confusion Matrix - Ensemble",
        os.path.join(config.output_dir, "confusion_matrix_ensemble.png")
    )

    plot_roc_curves(
        trained_models,
        X_test, y_test,
        os.path.join(config.output_dir, "roc_curves.png")
    )

    plot_metrics_comparison(
        metrics,
        os.path.join(config.output_dir, "metrics_comparison.png")
    )

    print("\nPavyzdinės įrašo lygio prognozės:")
    for rec in find_complete_records(config.data_dir)[:5]:
        result = predict_record_level(rec, trained_models["Ensemble"], config)
        print(result)

    print("\nBaigta.")
    print(f"\nGrafikai išsaugoti čia: {config.output_dir}")


if __name__ == "__main__":
    main()