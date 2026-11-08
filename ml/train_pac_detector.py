from __future__ import annotations

import ast
import json
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import wfdb
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "data" / "raw"
CACHE_ROOT = RAW_ROOT / "ptbxl_cache_lead_II"
DATABASE_PATH = RAW_ROOT / "ptbxl_database.csv"
OUTPUT_ROOT = ROOT / "models"
MODEL_PATH = OUTPUT_ROOT / "pac_detector.joblib"
METRICS_PATH = OUTPUT_ROOT / "pac_detector_metrics.json"
PTBXL_BASE = "ptb-xl/1.0.3"
FEATURE_LENGTH = 1000
RANDOM_SEED = 42
PAC_DECISION_THRESHOLD = 2.0


def load_signal(row: object) -> np.ndarray:
    ecg_id = int(row.ecg_id)
    cache_path = CACHE_ROOT / f"{ecg_id}.npz"
    if cache_path.exists():
        with np.load(cache_path) as cached:
            signal = cached["signal"].astype(np.float32)
            lead_names = tuple(str(name) for name in cached["lead_names"])
        if signal.shape == (FEATURE_LENGTH, 1) and lead_names == ("II",):
            return signal[:, 0]

    record_path = Path(str(row.filename_lr))
    remote_directory = f"{PTBXL_BASE}/{record_path.parent.as_posix()}"
    for attempt in range(4):
        try:
            record = wfdb.rdrecord(record_path.name, pn_dir=remote_directory)
            break
        except Exception:
            if attempt == 3:
                raise
            delay = 2**attempt
            print(f"Temporary download error for ECG {ecg_id}; retrying in {delay}s")
            time.sleep(delay)

    if record.p_signal is None or record.p_signal.shape != (FEATURE_LENGTH, 12):
        raise ValueError(f"Unexpected low-resolution signal shape for ECG {ecg_id}.")
    lead_index = record.sig_name.index("II")
    signal = np.nan_to_num(
        record.p_signal[:, lead_index], nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        signal=signal[:, None],
        lead_names=np.asarray(["II"]),
        sampling_rate=np.asarray(record.fs),
    )
    return signal


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    centered = signal.astype(np.float32) - float(np.median(signal))
    return centered / (float(np.std(centered)) + 1e-6)


def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(f"PTB-XL metadata not found: {DATABASE_PATH}")

    metadata = pd.read_csv(DATABASE_PATH)
    rows = list(metadata.itertuples(index=False))
    codes_by_id = {
        int(row.ecg_id): set(ast.literal_eval(row.scp_codes or "{}"))
        for row in rows
    }
    pac_patients = {
        int(row.patient_id)
        for row in rows
        if "PAC" in codes_by_id[int(row.ecg_id)]
    }

    selected = []
    for row in rows:
        ecg_id = int(row.ecg_id)
        has_pac = "PAC" in codes_by_id[ecg_id]
        cached_non_pac = (CACHE_ROOT / f"{ecg_id}.npz").exists()
        patient_id = int(row.patient_id)
        if has_pac or (cached_non_pac and patient_id not in pac_patients):
            selected.append((row, has_pac))

    features = []
    labels = []
    patients = []
    record_ids = []
    for index, (row, has_pac) in enumerate(selected, start=1):
        features.append(normalize_signal(load_signal(row)))
        labels.append(int(has_pac))
        patients.append(int(row.patient_id))
        record_ids.append(int(row.ecg_id))
        if index % 25 == 0 or index == len(selected):
            print(f"Loaded {index}/{len(selected)} records")

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
        np.asarray(patients, dtype=np.int64),
        record_ids,
    )


def create_classifier():
    return make_pipeline(
        StandardScaler(),
        LinearSVC(class_weight="balanced", random_state=RANDOM_SEED, max_iter=20000),
    )


def evaluate(
    features: np.ndarray,
    labels: np.ndarray,
    patients: np.ndarray,
    record_ids: list[int],
) -> dict[str, object]:
    predictions = np.empty(labels.shape, dtype=np.int8)
    scores = np.empty(labels.shape, dtype=np.float32)
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_SEED)
    for train_indices, test_indices in splitter.split(features, labels, patients):
        classifier = create_classifier()
        classifier.fit(features[train_indices], labels[train_indices])
        predictions[test_indices] = classifier.predict(features[test_indices])
        scores[test_indices] = classifier.decision_function(features[test_indices])
    threshold_predictions = (scores >= PAC_DECISION_THRESHOLD).astype(np.int8)
    threshold_report = classification_report(
        labels,
        threshold_predictions,
        labels=[0, 1],
        target_names=["no_PAC", "PAC"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "input_type": "PTB-XL 100 Hz Lead II numerical signals",
        "target": "SCP code PAC present versus not recorded",
        "validation": "4-fold stratified group cross-validation by patient",
        "sample_count": len(labels),
        "patient_count": len(np.unique(patients)),
        "class_counts": {"no_PAC": int(np.count_nonzero(labels == 0)), "PAC": int(np.count_nonzero(labels == 1))},
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "pac_average_precision": float(average_precision_score(labels, scores)),
        "pac_candidate_threshold": PAC_DECISION_THRESHOLD,
        "threshold_balanced_accuracy": float(balanced_accuracy_score(labels, threshold_predictions)),
        "threshold_precision": float(threshold_report["PAC"]["precision"]),
        "threshold_pac_recall": float(threshold_report["PAC"]["recall"]),
        "confusion_matrix_labels": ["no_PAC", "PAC"],
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=["no_PAC", "PAC"],
            output_dict=True,
            zero_division=0,
        ),
        "record_ids": record_ids,
        "warning": (
            "PAC labels are ECG-level annotations and may co-occur with other diagnoses. "
            "Unannotated records are treated as no_PAC. Signal-only evaluation does not validate photo extraction or clinical performance."
        ),
    }


def main() -> None:
    features, labels, patients, record_ids = load_dataset()
    if len(np.unique(labels)) != 2:
        raise ValueError("Both PAC and no-PAC examples are required.")
    metrics = evaluate(features, labels, patients, record_ids)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    classifier = create_classifier()
    classifier.fit(features, labels)
    joblib.dump(
        {
            "classifier": classifier,
            "labels": ["no_PAC", "PAC"],
            "decision_threshold": PAC_DECISION_THRESHOLD,
            "warning": metrics["warning"],
        },
        MODEL_PATH,
    )

    print(f"Patient-grouped accuracy: {metrics['accuracy']:.3f}")
    print(f"Patient-grouped balanced accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"PAC average precision: {metrics['pac_average_precision']:.3f}")
    print(
        f"PAC candidate threshold {PAC_DECISION_THRESHOLD:.1f}: "
        f"balanced accuracy={metrics['threshold_balanced_accuracy']:.3f}, "
        f"precision={metrics['threshold_precision']:.3f}, recall={metrics['threshold_pac_recall']:.3f}"
    )
    print(f"Class counts: {metrics['class_counts']}")
    print("Confusion matrix (rows=true, columns=predicted; order no_PAC, PAC):")
    print(np.asarray(metrics["confusion_matrix"]))
    print(f"Metrics saved to {METRICS_PATH}")
    print(f"Model saved to {MODEL_PATH}")
    print(metrics["warning"])


if __name__ == "__main__":
    main()