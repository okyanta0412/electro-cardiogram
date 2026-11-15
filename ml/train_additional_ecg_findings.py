from __future__ import annotations

import ast
import json
import random
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import wfdb
from sklearn.metrics import average_precision_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "data" / "raw"
CACHE_ROOT = RAW_ROOT / "ptbxl_cache_lead_II"
DATABASE_PATH = RAW_ROOT / "ptbxl_database.csv"
OUTPUT_ROOT = ROOT / "models"
MODEL_PATH = OUTPUT_ROOT / "additional_ecg_findings.joblib"
METRICS_PATH = OUTPUT_ROOT / "additional_ecg_findings_metrics.json"
PTBXL_BASE = "ptb-xl/1.0.3"
INPUT_LENGTH = 1000
TRAIN_FOLDS = set(range(1, 9))
TEST_FOLDS = {9, 10}
TARGETS = {
    "second_degree_av_block": "2AVB",
    "third_degree_av_block": "3AVB",
    "nonspecific_st_elevation": "STE_",
    "nonspecific_st_depression": "STD_",
}
RANDOM_SEED = 42
MAX_TRAIN_POSITIVES = 500
MAX_TEST_POSITIVES = 100
MAX_NEGATIVES_PER_SPLIT = 300


def load_signal(row: object) -> np.ndarray:
    ecg_id = int(row.ecg_id)
    cache_path = CACHE_ROOT / f"{ecg_id}.npz"
    if cache_path.exists():
        with np.load(cache_path) as cached:
            signal = cached["signal"].astype(np.float32)
            lead_names = tuple(str(name) for name in cached["lead_names"])
        if signal.shape == (INPUT_LENGTH, 1) and lead_names == ("II",):
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

    if record.p_signal is None or record.p_signal.shape != (INPUT_LENGTH, 12):
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
    centered = np.asarray(signal, dtype=np.float32).reshape(-1)
    centered = centered - float(np.median(centered))
    return centered / (float(np.std(centered)) + 1e-6)


def select_cohort(metadata: pd.DataFrame, code: str, randomizer: random.Random):
    rows = list(metadata.itertuples(index=False))
    codes_by_id = {int(row.ecg_id): set(ast.literal_eval(row.scp_codes or "{}")) for row in rows}
    positive_patients = {
        int(row.patient_id) for row in rows if code in codes_by_id[int(row.ecg_id)]
    }
    selected: dict[str, list[tuple[object, int]]] = {"train": [], "test": []}

    for split_name, folds, positive_limit in (
        ("train", TRAIN_FOLDS, MAX_TRAIN_POSITIVES),
        ("test", TEST_FOLDS, MAX_TEST_POSITIVES),
    ):
        positive_by_patient: dict[int, object] = {}
        negative_by_patient: dict[int, object] = {}
        for row in rows:
            patient_id = int(row.patient_id)
            if int(row.strat_fold) not in folds:
                continue
            codes = codes_by_id[int(row.ecg_id)]
            if code in codes:
                positive_by_patient.setdefault(patient_id, row)
            elif patient_id not in positive_patients and (CACHE_ROOT / f"{int(row.ecg_id)}.npz").exists():
                negative_by_patient.setdefault(patient_id, row)

        positive_rows = list(positive_by_patient.values())
        randomizer.shuffle(positive_rows)
        positive_rows = positive_rows[:positive_limit]
        negative_rows = list(negative_by_patient.values())
        randomizer.shuffle(negative_rows)
        negative_limit = min(MAX_NEGATIVES_PER_SPLIT, max(100, 3 * len(positive_rows)))
        negative_rows = negative_rows[:negative_limit]
        selected[split_name] = [(row, 1) for row in positive_rows]
        selected[split_name].extend((row, 0) for row in negative_rows)

    return selected, len(positive_patients)


def create_classifier():
    return make_pipeline(
        StandardScaler(),
        LinearSVC(class_weight="balanced", random_state=RANDOM_SEED, max_iter=20000),
    )


def train_detector(
    metadata: pd.DataFrame,
    code: str,
    randomizer: random.Random,
) -> tuple[object, dict[str, object]]:
    cohort, total_positive_patients = select_cohort(metadata, code, randomizer)
    split_data = {}
    for split_name in ("train", "test"):
        records = cohort[split_name]
        if not any(label for _, label in records) or not any(not label for _, label in records):
            raise ValueError(f"{code}: both positive and negative {split_name} examples are required.")
        features = []
        labels = []
        patient_ids = []
        ecg_ids = []
        for row, label in records:
            signal = load_signal(row)
            features.append(normalize_signal(signal))
            labels.append(label)
            patient_ids.append(int(row.patient_id))
            ecg_ids.append(int(row.ecg_id))
        split_data[split_name] = (
            np.asarray(features, dtype=np.float32),
            np.asarray(labels, dtype=np.int8),
            np.asarray(patient_ids, dtype=np.int64),
            ecg_ids,
        )

    train_features, train_labels, train_patients, train_ids = split_data["train"]
    test_features, test_labels, test_patients, test_ids = split_data["test"]
    if set(train_patients) & set(test_patients):
        raise ValueError(f"{code}: patient overlap between train and test splits.")

    model = create_classifier()
    model.fit(train_features, train_labels)
    predictions = model.predict(test_features)
    scores = model.decision_function(test_features)
    metrics = {
        "source_code": code,
        "validation": "official PTB-XL folds 1-8 train, folds 9-10 held out by patient",
        "total_positive_patients_in_metadata": total_positive_patients,
        "train_samples": len(train_labels),
        "test_samples": len(test_labels),
        "train_patient_count": len(np.unique(train_patients)),
        "test_patient_count": len(np.unique(test_patients)),
        "train_label_counts": dict(Counter(train_labels.tolist())),
        "test_label_counts": dict(Counter(test_labels.tolist())),
        "balanced_accuracy": float(balanced_accuracy_score(test_labels, predictions)),
        "average_precision": float(average_precision_score(test_labels, scores)),
        "confusion_matrix_labels": ["not_recorded", code],
        "confusion_matrix": confusion_matrix(test_labels, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            test_labels,
            predictions,
            labels=[0, 1],
            target_names=["not_recorded", code],
            output_dict=True,
            zero_division=0,
        ),
        "train_ecg_ids": train_ids,
        "test_ecg_ids": test_ids,
        "warning": (
            "Experimental numerical-signal detector only. PTB-XL ECG-level codes may coexist with other findings; "
            "uncoded records are treated as negatives. This does not validate photo extraction or clinical use."
        ),
    }
    return model, metrics


def main() -> None:
    metadata = pd.read_csv(DATABASE_PATH)
    randomizer = random.Random(RANDOM_SEED)
    models = {}
    metrics: dict[str, object] = {
        "input": "PTB-XL Lead II numerical signals, 100 Hz, 10 seconds",
        "tasks": {},
        "not_trained": {
            "ventricular_tachycardia": {
                "mitdb_annotated_episodes": 1,
                "reason": "Only one locally annotated VT episode; insufficient for a patient-held-out detector.",
            }
        },
    }

    for name, code in TARGETS.items():
        model, task_metrics = train_detector(metadata, code, randomizer)
        models[name] = model
        metrics["tasks"][name] = task_metrics
        print(
            f"{name} ({code}): balanced accuracy={task_metrics['balanced_accuracy']:.3f}, "
            f"average precision={task_metrics['average_precision']:.3f}, "
            f"train={task_metrics['train_samples']}, test={task_metrics['test_samples']}"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Models saved to {MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()