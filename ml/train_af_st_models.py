from __future__ import annotations

import ast
import json
import random
import time
import urllib.request
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import wfdb
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = RAW_DIR / "ptbxl_cache"
OUTPUT_DIR = ROOT / "models"
METADATA_URL = "https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv"
STATEMENTS_URL = "https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv"
DATABASE_PATH = RAW_DIR / "ptbxl_database.csv"
STATEMENTS_PATH = RAW_DIR / "scp_statements.csv"
MODELS_PATH = OUTPUT_DIR / "af_st_signal_models.joblib"
METRICS_PATH = OUTPUT_DIR / "af_st_signal_metrics.json"
MODEL_TASKS = {"AFIB": "AFIB", "STTC": "STTC"}
MAX_TRAIN_POSITIVES = 20
MAX_TEST_POSITIVES = 10
MAX_TRAIN_NEGATIVES = 30
MAX_TEST_NEGATIVES = 15
RANDOM_SEED = 42
MIN_METADATA_ROWS = 20000


def download_if_missing(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {path.name}...")
    urllib.request.urlretrieve(url, path)


def parse_scp_codes(value: str) -> set[str]:
    if not value or value == "{}":
        return set()
    return set(ast.literal_eval(value))


def load_labeled_metadata() -> tuple[pd.DataFrame, set[str]]:
    download_if_missing(METADATA_URL, DATABASE_PATH)
    download_if_missing(STATEMENTS_URL, STATEMENTS_PATH)

    metadata = pd.read_csv(DATABASE_PATH, on_bad_lines="skip")
    statements = pd.read_csv(STATEMENTS_PATH, index_col=0)
    sttc_codes = set(statements.index[statements["diagnostic_class"] == "STTC"])
    codes_by_row = [parse_scp_codes(value) for value in metadata["scp_codes"].fillna("{}")]
    metadata["has_AFIB"] = ["AFIB" in codes for codes in codes_by_row]
    metadata["has_STTC"] = [bool(codes & sttc_codes) for codes in codes_by_row]
    metadata["strat_fold"] = metadata["strat_fold"].astype(int)

    if len(metadata) < MIN_METADATA_ROWS:
        raise RuntimeError(f"Only {len(metadata)} metadata rows are available; PTB-XL metadata appears incomplete.")
    if metadata["has_AFIB"].sum() < 20 or metadata["has_STTC"].sum() < 20:
        raise RuntimeError("Not enough AFIB/STTC labels are available to form a train/test prototype.")
    return metadata, sttc_codes


def sample_task_records(
    metadata: pd.DataFrame,
    target: str,
    train_folds: set[int],
    test_folds: set[int],
    randomizer: random.Random,
) -> dict[int, tuple[int, int]]:
    selected: dict[int, tuple[int, int]] = {}
    is_positive = metadata[f"has_{target}"]

    for split_name, folds, positive_limit, negative_limit in (
        ("train", train_folds, MAX_TRAIN_POSITIVES, MAX_TRAIN_NEGATIVES),
        ("test", test_folds, MAX_TEST_POSITIVES, MAX_TEST_NEGATIVES),
    ):
        fold_rows = metadata[metadata["strat_fold"].isin(folds)]
        positives = fold_rows[is_positive.loc[fold_rows.index]]
        negatives = fold_rows[~is_positive.loc[fold_rows.index]]
        positive_ids = randomizer.sample(positives.index.tolist(), min(positive_limit, len(positives)))
        negative_ids = randomizer.sample(negatives.index.tolist(), min(negative_limit, len(negatives)))

        if not positive_ids or not negative_ids:
            raise RuntimeError(f"Could not select positive and negative {target} records for {split_name}.")

        for row_index in positive_ids:
            ecg_id = int(metadata.at[row_index, "ecg_id"])
            selected[ecg_id] = (row_index, 1)
        for row_index in negative_ids:
            ecg_id = int(metadata.at[row_index, "ecg_id"])
            selected[ecg_id] = (row_index, 0)

        print(f"{target} {split_name}: {len(positive_ids)} positive, {len(negative_ids)} negative")

    return selected


def load_signal(row: pd.Series) -> np.ndarray:
    ecg_id = int(row.ecg_id)
    cache_path = CACHE_DIR / f"{ecg_id}.npz"
    if cache_path.exists():
        with np.load(cache_path) as cached:
            return cached["signal"].astype(np.float32)

    record_path = Path(str(row.filename_lr))
    remote_directory = f"ptb-xl/1.0.3/{record_path.parent.as_posix()}"
    for attempt in range(4):
        try:
            record = wfdb.rdrecord(record_path.name, pn_dir=remote_directory)
            break
        except Exception:
            if attempt == 3:
                raise
            delay = 2 ** attempt
            print(f"Temporary download error for ECG {ecg_id}; retrying in {delay}s")
            time.sleep(delay)
    if record.p_signal is None or record.p_signal.shape != (1000, 12):
        raise RuntimeError(f"Unexpected waveform shape for ECG record {ecg_id}.")

    signal = np.nan_to_num(record.p_signal, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, signal=signal)
    return signal


def build_features(
    metadata: pd.DataFrame,
    selection: dict[int, tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row_by_id = {int(row.ecg_id): row for row in metadata.itertuples(index=False)}
    record_ids = sorted(selection)
    signals = []
    labels = []
    folds = []

    for index, ecg_id in enumerate(record_ids, start=1):
        row = row_by_id[ecg_id]
        signal = load_signal(row)
        signals.append(signal.reshape(-1))
        _, label = selection[ecg_id]
        labels.append(label)
        folds.append(int(row.strat_fold))
        if index % 25 == 0 or index == len(record_ids):
            print(f"Loaded {index}/{len(record_ids)} records")

    return (
        np.asarray(signals, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
        np.asarray(folds),
        np.asarray(record_ids, dtype=np.int64),
    )


def evaluate_task(
    target: str,
    metadata: pd.DataFrame,
    selection: dict[int, tuple[int, int]],
) -> tuple[object, dict[str, object]]:
    features, labels, folds, ecg_ids = build_features(metadata, selection)
    train_mask = folds <= 8
    test_mask = folds >= 9
    if len(set(labels[train_mask])) < 2 or len(set(labels[test_mask])) < 2:
        raise RuntimeError(f"Both labels must exist in train and test folds for {target}.")

    model = make_pipeline(
        StandardScaler(),
        LinearSVC(class_weight="balanced", random_state=RANDOM_SEED, max_iter=10000),
    )
    model.fit(features[train_mask], labels[train_mask])
    predictions = model.predict(features[test_mask])
    report = classification_report(
        labels[test_mask],
        predictions,
        labels=[0, 1],
        target_names=[f"not_{target}", target],
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "positive_label": target,
        "signal_input_shape": [1000, 12],
        "train_samples": int(train_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "train_ecg_ids": ecg_ids[train_mask].tolist(),
        "test_ecg_ids": ecg_ids[test_mask].tolist(),
        "train_label_counts": dict(Counter(labels[train_mask].tolist())),
        "test_label_counts": dict(Counter(labels[test_mask].tolist())),
        "balanced_accuracy": float(balanced_accuracy_score(labels[test_mask], predictions)),
        "classification_report": report,
        "confusion_matrix_labels": [f"not_{target}", target],
        "confusion_matrix": confusion_matrix(labels[test_mask], predictions, labels=[0, 1]).tolist(),
    }
    return model, metrics


def main() -> None:
    randomizer = random.Random(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata, _ = load_labeled_metadata()
    train_folds = set(range(1, 9))
    test_folds = {9, 10}
    models = {}
    results = {
        "dataset": "PTB-XL 1.0.3",
        "input": "12-lead 100 Hz numerical signals, 10 seconds",
        "train_folds": sorted(train_folds),
        "held_out_folds": sorted(test_folds),
        "metadata_rows_used": int(len(metadata)),
        "tasks": {},
        "warning": "Research prototype only. Not trained on photographs and not for clinical use.",
    }

    for target in MODEL_TASKS:
        selection = sample_task_records(metadata, target, train_folds, test_folds, randomizer)
        model, metrics = evaluate_task(target, metadata, selection)
        models[target] = model
        results["tasks"][target] = metrics

    joblib.dump(models, MODELS_PATH)
    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Models saved to {MODELS_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()
