from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from digitize_monitor_photo import (
    extract_trace,
    find_trace_bands,
    make_trace_mask,
)


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
CACHE_ROOT = RAW_ROOT / "ptbxl_cache_lead_II"
DATABASE_PATH = RAW_ROOT / "ptbxl_database.csv"
MODEL_PATH = ROOT / "models" / "monitor_photo_classifier.joblib"
METRICS_PATH = ROOT / "models" / "monitor_photo_classifier_metrics.json"
LABELS = ("NORMAL", "AF", "PVC")
FEATURE_LENGTH = 1000
RANDOM_SEED = 42


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    signal = signal - float(np.median(signal))
    return signal / (float(np.std(signal)) + 1e-6)


def extract_features_from_image(image: np.ndarray) -> np.ndarray:
    mask = make_trace_mask(image)
    bands = find_trace_bands(mask)
    y0, y1 = bands[0] if len(bands) == 1 else bands[1]
    x_values, voltage_pixels, _, _ = extract_trace(mask, y0, y1)
    sample_x = np.linspace(float(x_values.min()), float(x_values.max()), FEATURE_LENGTH)
    waveform = np.interp(sample_x, x_values, voltage_pixels)
    return normalize_signal(waveform)


def extract_features(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read monitor image: {image_path}")
    return extract_features_from_image(image)


def label_record(codes: set[str]) -> str | None:
    if "PVC" in codes:
        return "PVC"
    if codes & {"AFIB", "AFLT"}:
        return "AF"
    if "NORM" in codes:
        return "NORMAL"
    return None


def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, str]]]:
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(f"PTB-XL metadata is missing under {RAW_ROOT}")

    features = []
    labels = []
    groups = []
    records = []
    metadata = pd.read_csv(DATABASE_PATH)

    for row in metadata.itertuples(index=False):
        ecg_id = int(row.ecg_id)
        cache_path = CACHE_ROOT / f"{ecg_id}.npz"
        if not cache_path.exists():
            continue
        codes = set(ast.literal_eval(row.scp_codes or "{}"))
        label = label_record(codes)
        if label is None:
            continue
        with np.load(cache_path) as cached:
            signal = cached["signal"].astype(np.float32)
            lead_names = tuple(str(name) for name in cached["lead_names"])
        if signal.shape != (FEATURE_LENGTH, 1) or lead_names != ("II",):
            continue

        features.append(normalize_signal(signal))
        labels.append(label)
        groups.append(int(row.patient_id))
        records.append({"ecg_id": str(ecg_id), "patient_id": str(row.patient_id), "label": label})

    if not features:
        raise ValueError(f"No supported labeled Lead II signals found under {CACHE_ROOT}")
    return np.vstack(features), np.asarray(labels), np.asarray(groups), records


def create_classifier():
    return make_pipeline(
        StandardScaler(),
        LinearSVC(class_weight="balanced", random_state=RANDOM_SEED, max_iter=20000),
    )


def evaluate(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    records: list[dict[str, str]],
) -> dict[str, object]:
    predictions = np.empty(labels.shape, dtype=object)
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_SEED)
    for train_indices, test_indices in splitter.split(features, labels, groups):
        classifier = create_classifier()
        classifier.fit(features[train_indices], labels[train_indices])
        predictions[test_indices] = classifier.predict(features[test_indices])

    return {
        "input_type": "PTB-XL Lead II numerical signals; photo waveform extraction is not part of this evaluation",
        "validation": "4-fold stratified group cross-validation by patient",
        "sample_count": len(labels),
        "patient_count": len(np.unique(groups)),
        "feature_length": FEATURE_LENGTH,
        "labels": list(LABELS),
        "label_counts": {label: int(np.count_nonzero(labels == label)) for label in LABELS},
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=LABELS).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "predictions": [
            {**record, "predicted_label": str(prediction)}
            for record, prediction in zip(records, predictions)
        ],
        "warning": (
            "The PVC class has only eight records. These signal-only metrics do not validate photo digitization "
            "or clinical performance."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a three-class Lead II signal baseline for the monitor-photo demo."
    )
    parser.add_argument("--image", type=Path, help="Classify one high-contrast single-lead monitor image using the saved model.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    args = parser.parse_args()

    if args.image is not None:
        if not args.model.exists():
            raise SystemExit(f"Model not found: {args.model}. Run this script without --image first.")
        artifact = joblib.load(args.model)
        features = extract_features(args.image).reshape(1, -1)
        prediction = str(artifact["classifier"].predict(features)[0])
        print(json.dumps({"predicted_label": prediction, "warning": artifact["warning"]}, indent=2))
        return

    features, labels, groups, records = load_dataset()
    metrics = evaluate(features, labels, groups, records)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    classifier = create_classifier()
    classifier.fit(features, labels)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": classifier, "labels": LABELS, "warning": metrics["warning"]}, args.model)

    print(f"4-fold patient-grouped accuracy: {metrics['accuracy']:.3f}")
    print(f"4-fold patient-grouped balanced accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"Samples: {metrics['sample_count']} across {metrics['patient_count']} patients")
    print(f"Class counts: {metrics['label_counts']}")
    print("Confusion matrix (rows=true, columns=predicted; order NORMAL, AF, PVC):")
    print(np.asarray(metrics["confusion_matrix"]))
    print(f"Metrics saved to {args.metrics}")
    print(f"Demo model saved to {args.model}")
    print(metrics["warning"])


if __name__ == "__main__":
    main()