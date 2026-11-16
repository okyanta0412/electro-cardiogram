from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import joblib
import numpy as np
import pillow_heif
from PIL import Image, ImageOps
from sklearn.neighbors import NearestNeighbors

from digitize_monitor_photo import extract_trace
from monitor_photo_classifier import normalize_signal

pillow_heif.register_heif_opener()


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "labeled_photos"
MODEL_PATH = ROOT / "models" / "labeled_photo_prototype.joblib"
METRICS_PATH = ROOT / "models" / "labeled_photo_prototype_metrics.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
TRACE_DARK_THRESHOLD = 60
FEATURE_LENGTH = 1000


def extract_labeled_photo_waveform(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, TRACE_DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    x_values, voltage_pixels, _, _ = extract_trace(mask, 0, gray.shape[0])
    sample_x = np.linspace(float(x_values.min()), float(x_values.max()), FEATURE_LENGTH)
    waveform = np.interp(sample_x, x_values, voltage_pixels)
    return normalize_signal(waveform)


def load_labeled_photos() -> tuple[np.ndarray, np.ndarray, list[str]]:
    waveforms = []
    labels = []
    sample_paths = []
    for label_dir in sorted(path for path in DATA_ROOT.iterdir() if path.is_dir()):
        for image_path in sorted(
            path for path in label_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            with Image.open(image_path) as source:
                rgb_image = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
            image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            waveforms.append(extract_labeled_photo_waveform(image))
            labels.append(label_dir.name)
            sample_paths.append(image_path.relative_to(ROOT.parent).as_posix())

    if not waveforms:
        raise ValueError(f"No labeled photos found under {DATA_ROOT}")
    return np.vstack(waveforms), np.asarray(labels), sample_paths


def predict_labeled_photo(image: np.ndarray, artifact: dict[str, object]) -> dict[str, object]:
    waveform = extract_labeled_photo_waveform(image).reshape(1, -1)
    distances, neighbors = artifact["model"].kneighbors(waveform, n_neighbors=1)
    sample_index = int(neighbors[0, 0])
    return {
        "label": str(artifact["labels"][sample_index]),
        "cosine_distance": float(distances[0, 0]),
        "reference_image": str(artifact["sample_paths"][sample_index]),
    }


def train() -> dict[str, object]:
    waveforms, labels, sample_paths = load_labeled_photos()
    model = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
    model.fit(waveforms)
    distances, neighbors = model.kneighbors(waveforms)
    predictions = labels[neighbors[:, 0]]
    metrics: dict[str, object] = {
        "model": "one-nearest-neighbor cosine waveform prototype",
        "input": "HEIC/JPEG/PNG labeled ECG graph photos; grayscale pixels below threshold 60",
        "sample_count": len(labels),
        "class_count": len(np.unique(labels)),
        "samples_per_class": dict(sorted(Counter(labels.tolist()).items())),
        "resubstitution_accuracy": float(np.mean(predictions == labels)),
        "max_training_cosine_distance": float(np.max(distances)),
        "independent_validation": False,
        "warning": (
            "One reference image per current class. Resubstitution accuracy only verifies loading and exact-template matching; "
            "generalization to new images is not evaluated."
        ),
    }
    artifact = {
        "model": model,
        "labels": labels,
        "sample_paths": sample_paths,
        "metrics": metrics,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Model saved to {MODEL_PATH}")
    return artifact


def main() -> None:
    train()


if __name__ == "__main__":
    main()