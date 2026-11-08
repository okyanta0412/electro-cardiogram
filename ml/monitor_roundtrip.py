from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
OUTPUT_ROOT = DATA_ROOT / "threelead_monitor_roundtrip"
IMAGE_ROOT = OUTPUT_ROOT / "images"
METRICS_PATH = OUTPUT_ROOT / "metrics.json"
LEADS = ("II",)
IMAGE_WIDTH = 1200
IMAGE_HEIGHT = 460
PLOT_X0 = 120
PLOT_X1 = 1160
PLOT_TOP = 80
ROW_HEIGHT = 320
SWEEP_SECONDS = 10.0
SWEEP_SPEED_MM_PER_SECOND = 25.0
GAIN_MM_PER_MILLIVOLT = 10.0
SOURCE_SAMPLE_RATE_HZ = 500
PIXELS_PER_MM = (PLOT_X1 - PLOT_X0) / (SWEEP_SECONDS * SWEEP_SPEED_MM_PER_SECOND)
PIXELS_PER_SECOND = PIXELS_PER_MM * SWEEP_SPEED_MM_PER_SECOND
PIXELS_PER_MILLIVOLT = PIXELS_PER_MM * GAIN_MM_PER_MILLIVOLT
PLOT_BOTTOM = PLOT_TOP + ROW_HEIGHT * len(LEADS)


def find_pair_root() -> Path:
    candidates = (
        DATA_ROOT / "paired_12lead_photo_samples",
        DATA_ROOT / "paired_photo_samples",
    )
    for candidate in candidates:
        if (candidate / "manifest.csv").exists():
            return candidate
    raise FileNotFoundError("Run prepare_photo_signal_pairs.py first to create paired numeric signals.")


def render_monitor(signal: np.ndarray, lead_names: list[str], image_path: Path) -> None:
    lead_indices = [lead_names.index(lead) for lead in LEADS]
    canvas = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), (12, 18, 17), dtype=np.uint8)
    cv2.putText(
        canvas,
        "TEST MONITOR   LEAD II   25 mm/s   10 mm/mV   10 s",
        (PLOT_X0, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (185, 205, 195),
        1,
        cv2.LINE_AA,
    )

    total_grid_mm = int(round(SWEEP_SECONDS * SWEEP_SPEED_MM_PER_SECOND))
    for mm in range(total_grid_mm + 1):
        x = int(round(PLOT_X0 + mm * PIXELS_PER_MM))
        color = (35, 54, 45) if mm % 5 == 0 else (23, 36, 30)
        cv2.line(canvas, (x, PLOT_TOP), (x, PLOT_BOTTOM), color, 1)

    max_y_mm = int((ROW_HEIGHT / 2 - 5) / PIXELS_PER_MM)
    center = PLOT_TOP + ROW_HEIGHT // 2
    for mm in range(-max_y_mm, max_y_mm + 1):
        y = int(round(center - mm * PIXELS_PER_MM))
        color = (35, 54, 45) if mm % 5 == 0 else (23, 36, 30)
        cv2.line(canvas, (PLOT_X0, y), (PLOT_X1, y), color, 1)

    x_values = PLOT_X0 + np.arange(signal.shape[0]) / SOURCE_SAMPLE_RATE_HZ * PIXELS_PER_SECOND
    for row, (lead, column) in enumerate(zip(LEADS, lead_indices)):
        center = PLOT_TOP + row * ROW_HEIGHT + ROW_HEIGHT // 2
        values = signal[:, column].astype(np.float32)
        centered = values - float(np.median(values))
        if not np.isfinite(centered).all():
            raise ValueError(f"Lead {lead} contains invalid samples.")
        if float(np.max(np.abs(centered))) * PIXELS_PER_MILLIVOLT >= ROW_HEIGHT / 2 - 5:
            raise ValueError(f"Lead {lead} would clip at the fixed 10 mm/mV display gain.")
        y_values = center - centered * PIXELS_PER_MILLIVOLT
        points = np.column_stack((x_values, y_values)).round().astype(np.int32)
        cv2.polylines(canvas, [points], False, (95, 245, 150), 2, cv2.LINE_AA)
        cv2.putText(canvas, lead, (35, center + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 215, 190), 1, cv2.LINE_AA)

    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), canvas):
        raise OSError(f"Could not write monitor mockup: {image_path}")


def extract_monitor(image_path: Path, record_id: int, lead_names: list[str], source: np.ndarray) -> list[dict[str, object]]:
    image = cv2.imread(str(image_path))
    if image is None or image.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError(f"Unexpected monitor image dimensions for ECG {record_id}.")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    trace_mask = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 90) & (hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 90)
    x_values = np.arange(PLOT_X0, PLOT_X1 + 1)
    sample_times = (x_values - PLOT_X0) / PIXELS_PER_SECOND
    reference_times = np.arange(source.shape[0]) / SOURCE_SAMPLE_RATE_HZ
    results = []

    for row, lead in enumerate(LEADS):
        column = lead_names.index(lead)
        center = PLOT_TOP + row * ROW_HEIGHT + ROW_HEIGHT // 2
        y_min = max(0, int(center - ROW_HEIGHT * 0.48))
        y_max = min(IMAGE_HEIGHT, int(center + ROW_HEIGHT * 0.48))
        estimated = np.full(len(x_values), np.nan, dtype=np.float32)
        for index, x in enumerate(x_values):
            candidates = np.flatnonzero(trace_mask[y_min:y_max, x]) + y_min
            if candidates.size:
                estimated[index] = float(np.median(candidates))

        valid = np.isfinite(estimated)
        if valid.mean() < 0.9:
            raise RuntimeError(f"Trace coverage for ECG {record_id} lead {lead} is only {valid.mean():.1%}.")
        estimated = np.interp(np.arange(len(estimated)), np.flatnonzero(valid), estimated[valid])
        reconstructed = (center - estimated) / PIXELS_PER_MILLIVOLT
        reference = source[:, column].astype(np.float32)
        reference = reference - np.median(reference)
        reference = np.interp(sample_times, reference_times, reference)
        reconstructed = reconstructed - np.mean(reconstructed)
        reference = reference - np.mean(reference)
        corr = float(np.corrcoef(reconstructed, reference)[0, 1])
        rmse = float(np.sqrt(np.mean((reconstructed - reference) ** 2)))
        results.append({"lead": lead, "trace_coverage": float(valid.mean()), "correlation": corr, "rmse_mV": rmse})
    return results


def main() -> None:
    pair_root = find_pair_root()
    rows = list(csv.DictReader((pair_root / "manifest.csv").open(encoding="utf-8")))
    metrics = {
        "input_type": "synthetic single-lead bedside-monitor rendering from paired numerical signals",
        "leads": list(LEADS),
        "sampling_rate_hz": SOURCE_SAMPLE_RATE_HZ,
        "sweep_speed_mm_per_second": SWEEP_SPEED_MM_PER_SECOND,
        "gain_mm_per_millivolt": GAIN_MM_PER_MILLIVOLT,
        "sweep_seconds": SWEEP_SECONDS,
        "pixels_per_second": PIXELS_PER_SECOND,
        "pixels_per_millivolt": PIXELS_PER_MILLIVOLT,
        "records": [],
        "warning": "This round-trip test does not validate phone-photo artifacts or real monitor screenshots.",
    }

    for row in rows:
        record_id = int(row["ecg_id"])
        with np.load(pair_root / row["signal_path"]) as data:
            signal = data["signal"].astype(np.float32)
            lead_names = [str(name) for name in data["lead_names"]]
        if signal.shape[1] != len(lead_names) or not all(lead in lead_names for lead in LEADS):
            raise ValueError(f"ECG {record_id} does not contain all required leads {LEADS}.")

        image_path = IMAGE_ROOT / row["label"] / f"{record_id}.png"
        render_monitor(signal, lead_names, image_path)
        lead_metrics = extract_monitor(image_path, record_id, lead_names, signal)
        metrics["records"].append({"ecg_id": record_id, "label": row["label"], "image": image_path.relative_to(OUTPUT_ROOT).as_posix(), "leads": lead_metrics})
        print(f"ECG {record_id} {row['label']}: " + ", ".join(f"{item['lead']} corr={item['correlation']:.3f} rmse={item['rmse_mV']:.3f}mV" for item in lead_metrics))

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Rendered {len(rows)} synthetic single-Lead-II images; metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
