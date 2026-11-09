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
from scipy.signal import find_peaks
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
RHYTHM_FEATURE_NAMES = (
    "qrs_peak_count",
    "mean_rr_seconds",
    "sdnn_seconds",
    "rr_coefficient_of_variation",
    "rmssd_seconds",
    "fraction_rr_changes_over_50ms",
    "median_peak_prominence",
    "peak_prominence_std",
    "signal_derivative_std",
    "absolute_amplitude_p95",
)
RANDOM_SEED = 42
PVC_AF_REVIEW_MARGIN = 2.0


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    signal = signal - float(np.median(signal))
    return signal / (float(np.std(signal)) + 1e-6)


def extract_waveform_from_image(image: np.ndarray) -> np.ndarray:
    mask = make_trace_mask(image)
    bands = find_trace_bands(mask)
    y0, y1 = bands[0] if len(bands) == 1 else bands[1]
    x_values, voltage_pixels, _, _ = extract_trace(mask, y0, y1)
    sample_x = np.linspace(float(x_values.min()), float(x_values.max()), FEATURE_LENGTH)
    waveform = np.interp(sample_x, x_values, voltage_pixels)
    return normalize_signal(waveform)


def extract_features_from_waveform(waveform: np.ndarray) -> np.ndarray:
    waveform = normalize_signal(waveform)
    peaks, properties = find_peaks(np.abs(waveform), distance=35, prominence=0.7)
    rr_intervals = np.diff(peaks) / 100.0
    rr_changes = np.diff(rr_intervals)
    prominences = properties.get("prominences", np.asarray([], dtype=np.float32))

    return np.asarray(
        [
            len(peaks),
            float(np.mean(rr_intervals)) if rr_intervals.size else 0.0,
            float(np.std(rr_intervals)) if rr_intervals.size else 0.0,
            float(np.std(rr_intervals) / (np.mean(rr_intervals) + 1e-6)) if rr_intervals.size else 0.0,
            float(np.sqrt(np.mean(rr_changes**2))) if rr_changes.size else 0.0,
            float(np.mean(np.abs(rr_changes) > 0.05)) if rr_changes.size else 0.0,
            float(np.median(prominences)) if prominences.size else 0.0,
            float(np.std(prominences)) if prominences.size else 0.0,
            float(np.std(np.diff(waveform))),
            float(np.percentile(np.abs(waveform), 95)),
        ],
        dtype=np.float32,
    )


def extract_features_from_image(image: np.ndarray) -> np.ndarray:
    return extract_features_from_waveform(extract_waveform_from_image(image))


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

    metadata = pd.read_csv(DATABASE_PATH)
    features = []
    labels = []
    groups = []
    records = []
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
        features.append(extract_features_from_waveform(signal))
        labels.append(label)
        groups.append(int(row.patient_id))
        records.append({"ecg_id": str(row.ecg_id), "patient_id": str(row.patient_id), "label": label})

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
    decision_scores = np.empty((len(labels), len(LABELS)), dtype=np.float32)
    class_order: np.ndarray | None = None
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_SEED)
    for train_indices, test_indices in splitter.split(features, labels, groups):
        classifier = create_classifier()
        classifier.fit(features[train_indices], labels[train_indices])
        class_order = classifier.named_steps["linearsvc"].classes_
        decision_scores[test_indices] = classifier.decision_function(features[test_indices])

    if class_order is None:
        raise ValueError("Could not fit a classifier for cross-validation.")
    raw_predictions = class_order[np.argmax(decision_scores, axis=1)]
    pvc_index = int(np.flatnonzero(class_order == "PVC")[0])
    af_index = int(np.flatnonzero(class_order == "AF")[0])
    pvc_af_margin = decision_scores[:, pvc_index] - decision_scores[:, af_index]
    review_mask = (raw_predictions == "PVC") & (pvc_af_margin < PVC_AF_REVIEW_MARGIN)
    predictions = raw_predictions.astype(object)
    predictions[review_mask] = "REVIEW"
    accepted_mask = ~review_mask

    return {
        "input_type": "PTB-XL Lead II numerical signals; photo waveform extraction is not part of this evaluation",
        "validation": "4-fold stratified group cross-validation by patient",
        "sample_count": len(labels),
        "patient_count": len(np.unique(groups)),
        "feature_names": list(RHYTHM_FEATURE_NAMES),
        "labels": list(LABELS),
        "label_counts": {label: int(np.count_nonzero(labels == label)) for label in LABELS},
        "accuracy": float(accuracy_score(labels, raw_predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, raw_predictions)),
        "pvc_af_review_margin": PVC_AF_REVIEW_MARGIN,
        "review_count": int(review_mask.sum()),
        "review_rate": float(review_mask.mean()),
        "accuracy_when_not_reviewed": float(accuracy_score(labels[accepted_mask], raw_predictions[accepted_mask])),
        "balanced_accuracy_when_not_reviewed": float(
            balanced_accuracy_score(labels[accepted_mask], raw_predictions[accepted_mask])
        ),
        "af_to_pvc_after_review": int(np.count_nonzero((labels == "AF") & (predictions == "PVC"))),
        "confusion_matrix_labels": [*LABELS, "REVIEW"],
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[*LABELS, "REVIEW"]).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=[*LABELS, "REVIEW"],
            output_dict=True,
            zero_division=0,
        ),
        "predictions": [
            {**record, "predicted_label": str(prediction)}
            for record, prediction in zip(records, predictions)
        ],
        "warning": (
            "These metrics use numerical ECG signals, not photo-derived waveforms. The PVC/AF review threshold is exploratory, "
            "selected on the same cross-validation predictions, and is not an independent performance estimate."
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
    joblib.dump(
        {
            "classifier": classifier,
            "labels": LABELS,
            "pvc_af_review_margin": PVC_AF_REVIEW_MARGIN,
            "warning": metrics["warning"],
        },
        args.model,
    )

    print(f"Patient-grouped accuracy: {metrics['accuracy']:.3f}")
    print(f"Patient-grouped balanced accuracy: {metrics['balanced_accuracy']:.3f}")
    print(
        f"PVC/AF review: {metrics['review_count']} ({metrics['review_rate']:.1%}) held for review; "
        f"AF-to-PVC remaining={metrics['af_to_pvc_after_review']}"
    )
    print(f"Accuracy when not held: {metrics['accuracy_when_not_reviewed']:.3f}")
    print(f"Samples: {metrics['sample_count']} across {metrics['patient_count']} patients")
    print(f"Class counts: {metrics['label_counts']}")
    print("Confusion matrix (rows=true, columns=predicted; order NORMAL, AF, PVC):")
    print(np.asarray(metrics["confusion_matrix"]))
    print(f"Metrics saved to {args.metrics}")
    print(f"Demo model saved to {args.model}")
    print(metrics["warning"])


if __name__ == "__main__":
    main()