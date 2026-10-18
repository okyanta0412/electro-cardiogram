from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from skimage.feature import hog


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "beat_classifier.joblib"
IMAGE_SIZE = (160, 96)


def extract_features(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as source:
        image = source.convert("L")
        if image.size != IMAGE_SIZE:
            raise ValueError(
                f"Expected a rendered single-beat image of {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} pixels; "
                "full ECG photographs and full strips are not supported by this prototype."
            )

        features = hog(
            np.asarray(image),
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            channel_axis=None,
        )
    return features.reshape(1, -1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify one 160x96 MIT-BIH-derived beat plot as NORMAL or PVC."
    )
    parser.add_argument("image", type=Path, help="Path to a rendered single-beat PNG")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}. Run train_model.py first.")

    classifier = joblib.load(MODEL_PATH)
    features = extract_features(args.image)
    label = str(classifier.predict(features)[0])
    margin = float(classifier.decision_function(features)[0])
    print(
        json.dumps(
            {
                "classification": label,
                "decision_margin": margin,
                "decision_margin_note": "Uncalibrated SVM margin; not a probability or clinical confidence.",
                "input": str(args.image),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
