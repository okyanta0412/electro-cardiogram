from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "af_st_signal_models.joblib"
EXPECTED_SHAPE = (1000, 12)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict AFIB and STTC for one PTB-XL-shaped 12-lead numerical ECG signal."
    )
    parser.add_argument("signal", type=Path, help="Path to a cached NPZ containing a 1000x12 'signal' array")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(f"Models not found: {MODEL_PATH}. Run train_af_st_models.py first.")

    with np.load(args.signal) as source:
        signal = source["signal"].astype(np.float32)
    if signal.shape != EXPECTED_SHAPE:
        raise SystemExit(f"Expected a {EXPECTED_SHAPE[0]}x{EXPECTED_SHAPE[1]} numerical signal, got {signal.shape}.")

    models = joblib.load(MODEL_PATH)
    features = signal.reshape(1, -1)
    results = {}
    for target, model in models.items():
        margin = float(model.decision_function(features)[0])
        results[target] = {
            "prediction": "positive" if margin > 0 else "negative",
            "decision_margin": margin,
            "note": "Uncalibrated SVM margin; not a probability or clinical confidence.",
        }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
