from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import wfdb
from PIL import Image, ImageDraw
from skimage.feature import hog
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC
import joblib


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw" / "mitdb"
OUTPUT_DIR = ROOT / "models"
SAMPLE_DIR = OUTPUT_DIR / "sample_images"

RECORD_IDS = ("105", "106", "108", "119")
TEST_RECORD_IDS = {"106", "108"}
BEAT_LABELS = {"N": "NORMAL", "V": "PVC"}
MAX_BEATS_PER_CLASS_PER_RECORD = 60
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 96
WINDOW_BEFORE_SECONDS = 0.28
WINDOW_AFTER_SECONDS = 0.44
RANDOM_SEED = 42


def ensure_recordings() -> None:
    required_files = [
        DATA_DIR / f"{record_id}.{extension}"
        for record_id in RECORD_IDS
        for extension in ("dat", "hea", "atr")
    ]
    if all(path.exists() for path in required_files):
        print(f"Found MIT-BIH recordings in {DATA_DIR}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading MIT-BIH Arrhythmia Database (~100 MB)...")
    wfdb.dl_database(
        "mitdb",
        dl_dir=str(DATA_DIR),
        records=list(RECORD_IDS),
        annotators=["atr"],
        keep_subdirs=False,
    )


def select_beats(record_id: str, randomizer: random.Random) -> tuple[list[np.ndarray], list[str]]:
    record_path = str(DATA_DIR / record_id)
    record = wfdb.rdrecord(record_path, channels=[0])
    annotation = wfdb.rdann(record_path, "atr")
    if record.p_signal is None:
        raise RuntimeError(f"No physical signal data in record {record_id}")

    signal = record.p_signal[:, 0]
    sample_rate = float(record.fs)
    before = int(round(WINDOW_BEFORE_SECONDS * sample_rate))
    after = int(round(WINDOW_AFTER_SECONDS * sample_rate))
    selected: dict[str, list[tuple[int, int]]] = {"NORMAL": [], "PVC": []}
    seen = Counter()

    for sample, symbol in zip(annotation.sample, annotation.symbol):
        label = BEAT_LABELS.get(symbol)
        if label is None or sample < before or sample + after >= len(signal):
            continue

        seen[label] += 1
        candidates = selected[label]
        entry = (int(sample - before), int(sample + after))
        if len(candidates) < MAX_BEATS_PER_CLASS_PER_RECORD:
            candidates.append(entry)
        else:
            replacement = randomizer.randrange(seen[label])
            if replacement < MAX_BEATS_PER_CLASS_PER_RECORD:
                candidates[replacement] = entry

    images: list[np.ndarray] = []
    labels: list[str] = []
    for label, windows in selected.items():
        randomizer.shuffle(windows)
        for start, end in windows:
            image = render_beat_image(signal[start:end], IMAGE_WIDTH, IMAGE_HEIGHT)
            images.append(np.asarray(image, dtype=np.uint8))
            labels.append(label)
            sample_path = SAMPLE_DIR / f"{label}.png"
            if not sample_path.exists():
                image.save(sample_path)

    print(f"Record {record_id}: selected {len(labels)} beats")
    return images, labels


def render_beat_image(signal: np.ndarray, width: int, height: int) -> Image.Image:
    paper = Image.new("RGB", (width, height), "#fffdfc")
    drawing = ImageDraw.Draw(paper)

    for x in range(0, width, 8):
        major = x % 40 == 0
        drawing.line((x, 0, x, height), fill="#dfbdb8" if major else "#f1dedb", width=1)
    for y in range(0, height, 8):
        major = y % 40 == 0
        drawing.line((0, y, width, y), fill="#dfbdb8" if major else "#f1dedb", width=1)

    centered = np.asarray(signal, dtype=np.float32)
    centered = centered - float(np.median(centered))
    scale = float(np.percentile(np.abs(centered), 98))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    normalized = np.clip(centered / scale, -1.0, 1.0)
    x_values = np.linspace(0, width - 1, num=len(normalized))
    y_values = (height - 1) / 2 - normalized * (height * 0.38)
    points = [(int(round(x)), int(round(y))) for x, y in zip(x_values, y_values)]
    drawing.line(points, fill="#273d38", width=2)
    return paper


def extract_features(images: list[np.ndarray]) -> np.ndarray:
    return np.asarray(
        [
            hog(
                np.asarray(Image.fromarray(image).convert("L")),
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
                channel_axis=None,
            )
            for image in images
        ],
        dtype=np.float32,
    )


def collect_split(record_ids: list[str], randomizer: random.Random) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[str] = []
    for record_id in record_ids:
        record_images, record_labels = select_beats(record_id, randomizer)
        images.extend(record_images)
        labels.extend(record_labels)
    if not images:
        raise RuntimeError("No labeled beats were found. Check the downloaded recordings and annotations.")
    return extract_features(images), np.asarray(labels)


def main() -> None:
    randomizer = random.Random(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    ensure_recordings()

    train_records = [record_id for record_id in RECORD_IDS if record_id not in TEST_RECORD_IDS]
    test_records = [record_id for record_id in RECORD_IDS if record_id in TEST_RECORD_IDS]
    train_features, train_labels = collect_split(train_records, randomizer)
    test_features, test_labels = collect_split(test_records, randomizer)

    expected_labels = set(BEAT_LABELS.values())
    if set(train_labels) != expected_labels or set(test_labels) != expected_labels:
        raise RuntimeError(
            "Both NORMAL and PVC beats must occur in train and held-out records. "
            f"Train={Counter(train_labels)}, test={Counter(test_labels)}"
        )

    classifier = LinearSVC(class_weight="balanced", random_state=RANDOM_SEED, max_iter=10000)
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)

    report = classification_report(
        test_labels,
        predictions,
        labels=["NORMAL", "PVC"],
        output_dict=True,
        zero_division=0,
    )
    results = {
        "task": "MIT-BIH annotated beat image prototype",
        "classes": {"NORMAL": "annotation N", "PVC": "annotation V"},
        "train_records": train_records,
        "held_out_records": test_records,
        "train_counts": dict(Counter(train_labels)),
        "test_counts": dict(Counter(test_labels)),
        "balanced_accuracy": float(balanced_accuracy_score(test_labels, predictions)),
        "classification_report": report,
        "confusion_matrix_labels": ["NORMAL", "PVC"],
        "confusion_matrix": confusion_matrix(test_labels, predictions, labels=["NORMAL", "PVC"]).tolist(),
        "warning": "Research prototype only. Does not classify full-strip photos or support clinical decisions.",
    }

    joblib.dump(classifier, OUTPUT_DIR / "beat_classifier.joblib")
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Model saved to {OUTPUT_DIR / 'beat_classifier.joblib'}")
    print(f"Rendered examples saved to {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
