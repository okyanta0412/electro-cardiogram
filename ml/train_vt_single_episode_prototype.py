from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import wfdb
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
MITDB_ROOT = ROOT / "data" / "raw" / "mitdb"
OUTPUT_ROOT = ROOT / "models"
MODEL_PATH = OUTPUT_ROOT / "vt_single_episode_prototype.joblib"
METRICS_PATH = OUTPUT_ROOT / "vt_single_episode_prototype_metrics.json"
SOURCE_RECORD = "106"
WINDOW_SECONDS = 1.0
POSITIVE_STEP_SECONDS = 0.1
NEGATIVE_STEP_SECONDS = 0.5
RANDOM_SEED = 42


def normalized_windows(segment: np.ndarray, window_size: int, step_size: int) -> list[np.ndarray]:
    windows = []
    for start in range(0, len(segment) - window_size + 1, step_size):
        window = np.asarray(segment[start : start + window_size], dtype=np.float32)
        centered = window - float(np.median(window))
        windows.append(centered / (float(np.std(centered)) + 1e-6))
    return windows


def main() -> None:
    record_path = MITDB_ROOT / SOURCE_RECORD
    record = wfdb.rdrecord(str(record_path))
    annotations = wfdb.rdann(str(record_path), extension="atr")
    if record.p_signal is None:
        raise ValueError(f"Record {SOURCE_RECORD} has no physical signal.")
    if "MLII" not in record.sig_name:
        raise ValueError(f"Record {SOURCE_RECORD} has no MLII lead.")

    signal = record.p_signal[:, record.sig_name.index("MLII")].astype(np.float32)
    rhythm_changes = [
        (int(sample), (note or "").strip().strip("\x00"))
        for sample, symbol, note in zip(annotations.sample, annotations.symbol, annotations.aux_note)
        if symbol == "+" and (note or "").strip().strip("\x00")
    ]
    vt_intervals = []
    normal_intervals = []
    for index, (start, rhythm) in enumerate(rhythm_changes):
        end = rhythm_changes[index + 1][0] if index + 1 < len(rhythm_changes) else len(signal)
        if rhythm == "(VT":
            vt_intervals.append((start, end))
        elif rhythm == "(N":
            normal_intervals.append((start, end))

    if len(vt_intervals) != 1:
        raise ValueError(f"Expected exactly one annotated VT episode in record {SOURCE_RECORD}; found {len(vt_intervals)}.")

    window_size = int(round(record.fs * WINDOW_SECONDS))
    positive_windows = []
    for start, end in vt_intervals:
        positive_windows.extend(
            normalized_windows(signal[start:end], window_size, max(1, int(record.fs * POSITIVE_STEP_SECONDS)))
        )

    negative_windows = []
    for start, end in normal_intervals:
        negative_windows.extend(
            normalized_windows(signal[start:end], window_size, max(1, int(record.fs * NEGATIVE_STEP_SECONDS)))
        )

    if not positive_windows or not negative_windows:
        raise ValueError("The annotated record did not produce both VT and normal training windows.")

    features = np.vstack([negative_windows, positive_windows])
    labels = np.asarray([0] * len(negative_windows) + [1] * len(positive_windows), dtype=np.int8)
    classifier = make_pipeline(
        StandardScaler(),
        LinearSVC(class_weight="balanced", random_state=RANDOM_SEED, max_iter=20000),
    )
    classifier.fit(features, labels)

    metrics = {
        "target": "VT-like window vs MIT-BIH normal-rhythm window",
        "source_record": SOURCE_RECORD,
        "source_lead": "MLII",
        "sampling_rate_hz": float(record.fs),
        "vt_episode_count": len(vt_intervals),
        "vt_episode_seconds": [
            [round(start / record.fs, 3), round(end / record.fs, 3)] for start, end in vt_intervals
        ],
        "window_seconds": WINDOW_SECONDS,
        "positive_overlapping_windows": len(positive_windows),
        "negative_normal_windows": len(negative_windows),
        "independent_validation": False,
        "warning": (
            "Experimental one-episode template classifier only. Positive windows overlap and all come from one patient/event. "
            "No independent validation; not connected to the public photo API and not for clinical use."
        ),
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": classifier, "metrics": metrics}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Prototype model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()