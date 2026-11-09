from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import correlate, correlation_lags


LEAD_NAME = "II"
TRACE_HSV_LOWER = (35, 120, 120)
TRACE_HSV_UPPER = (95, 255, 255)
MIN_BAND_HEIGHT = 20
MAX_ROW_GAP = 80
MIN_ROW_PIXELS = 35
SPIKE_MEDIAN_WINDOW = 9
SPIKE_DEVIATION_THRESHOLD_PX = 12.0
MAX_ISOLATED_SPIKE_WIDTH = 2
MAX_TRACE_MASK_DENSITY = 0.18


def make_trace_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, dark_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bright_density = float(np.mean(bright_mask > 0))
    dark_density = float(np.mean(dark_mask > 0))

    candidates = [
        (density, mask)
        for density, mask in ((bright_density, bright_mask), (dark_density, dark_mask))
        if 0 < density <= MAX_TRACE_MASK_DENSITY
    ]
    if not candidates:
        raise ValueError(
            "Could not isolate a high-contrast waveform from the background. "
            "Crop to one trace and reduce glare or grid visibility."
        )
    return min(candidates, key=lambda candidate: candidate[0])[1]


def find_trace_bands(mask: np.ndarray) -> list[tuple[int, int]]:
    row_counts = np.count_nonzero(mask, axis=1)
    active_rows = np.flatnonzero(row_counts >= MIN_ROW_PIXELS)
    if active_rows.size == 0:
        raise ValueError("No bright-green ECG trace found. Check image color and contrast.")

    bands: list[list[int]] = []
    for row in active_rows:
        if not bands or row - bands[-1][-1] > MAX_ROW_GAP:
            bands.append([int(row)])
        else:
            bands[-1].append(int(row))

    detected = [(band[0], band[-1] + 1) for band in bands if band[-1] - band[0] + 1 >= MIN_BAND_HEIGHT]
    if len(detected) not in (1, 3):
        raise ValueError(f"Expected one trace or three ordered I/II/III rows, found {len(detected)}: {detected}")
    return detected


def parse_plot_corners(value: str) -> np.ndarray:
    try:
        corners = np.asarray(
            [[float(coordinate) for coordinate in point.split(",")] for point in value.split(";")],
            dtype=np.float32,
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Use four x,y points in clockwise order: top-left;top-right;bottom-right;bottom-left."
        ) from error
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise argparse.ArgumentTypeError(
            "Use four finite x,y points in clockwise order: top-left;top-right;bottom-right;bottom-left."
        )

    edges = np.roll(corners, -1, axis=0) - corners
    next_edges = np.roll(edges, -1, axis=0)
    cross_products = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
    if np.any(np.abs(cross_products) < 1) or not (np.all(cross_products > 0) or np.all(cross_products < 0)):
        raise argparse.ArgumentTypeError("Plot corners must form a non-degenerate convex quadrilateral.")
    return corners


def rectify_plot(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    if (
        np.any(corners[:, 0] < 0)
        or np.any(corners[:, 0] >= image_width)
        or np.any(corners[:, 1] < 0)
        or np.any(corners[:, 1] >= image_height)
    ):
        raise ValueError(f"Plot corners must be inside the image dimensions {image_width}x{image_height}.")

    top_left, top_right, bottom_right, bottom_left = corners
    output_width = int(round(max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left))))
    output_height = int(round(max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right))))
    if output_width < 100 or output_height < 100:
        raise ValueError("The selected plot area is too small to rectify.")

    destination = np.asarray(
        [[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1], [0, output_height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(image, transform, (output_width, output_height), flags=cv2.INTER_CUBIC)


def register_to_reference(image: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, float, int, float]:
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.015)
    reference_keypoints, reference_descriptors = detector.detectAndCompute(reference_gray, None)
    image_keypoints, image_descriptors = detector.detectAndCompute(image_gray, None)
    if reference_descriptors is None or image_descriptors is None:
        raise ValueError("Could not find registration features in the reference image or photo.")

    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(reference_descriptors, image_descriptors, k=2)
    matches = [first for pair in pairs if len(pair) == 2 for first, second in [pair] if first.distance < 0.72 * second.distance]
    if len(matches) < 12:
        raise ValueError(f"Too few reference-image feature matches ({len(matches)}); use --plot-corners instead.")

    reference_points = np.float32([reference_keypoints[match.queryIdx].pt for match in matches])
    image_points = np.float32([image_keypoints[match.trainIdx].pt for match in matches])
    homography, inlier_mask = cv2.findHomography(reference_points, image_points, cv2.RANSAC, 5.0)
    if homography is None or inlier_mask is None:
        raise ValueError("Could not estimate a perspective transform; use --plot-corners instead.")

    inliers = inlier_mask.ravel().astype(bool)
    inlier_count = int(inliers.sum())
    if inlier_count < 12 or inlier_count / len(matches) < 0.25:
        raise ValueError(
            f"Reference-image registration is ambiguous ({inlier_count}/{len(matches)} inliers); "
            "use --plot-corners instead."
        )

    projected = cv2.perspectiveTransform(reference_points[inliers].reshape(-1, 1, 2), homography).reshape(-1, 2)
    reprojection_errors = np.linalg.norm(projected - image_points[inliers], axis=1)
    median_error = float(np.median(reprojection_errors))
    if median_error > 4.0:
        raise ValueError(f"Reference-image registration error is too high ({median_error:.1f}px).")

    reference_height, reference_width = reference.shape[:2]
    source_corners = np.float32(
        [[[0, 0], [reference_width - 1, 0], [reference_width - 1, reference_height - 1], [0, reference_height - 1]]]
    )
    projected_corners = cv2.perspectiveTransform(source_corners, homography)[0]
    edge_scales = [
        np.linalg.norm(projected_corners[1] - projected_corners[0]) / reference_width,
        np.linalg.norm(projected_corners[2] - projected_corners[3]) / reference_width,
        np.linalg.norm(projected_corners[3] - projected_corners[0]) / reference_height,
        np.linalg.norm(projected_corners[2] - projected_corners[1]) / reference_height,
    ]
    output_scale = float(np.mean(edge_scales))
    if not 0.5 <= output_scale <= 10.0:
        raise ValueError(f"Reference-image scale is implausible ({output_scale:.2f}x).")

    output_width = int(round(reference_width * output_scale))
    output_height = int(round(reference_height * output_scale))
    scale_transform = np.diag([output_scale, output_scale, 1.0])
    photo_to_output = scale_transform @ np.linalg.inv(homography)
    registered = cv2.warpPerspective(
        image,
        photo_to_output,
        (output_width, output_height),
        flags=cv2.INTER_CUBIC,
    )
    return registered, output_scale, inlier_count, median_error


def extract_trace(mask: np.ndarray, y0: int, y1: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    row_mask = mask[y0:y1]
    row_mask = cv2.morphologyEx(
        row_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1)),
    )
    x_values = np.flatnonzero(np.count_nonzero(row_mask, axis=0) > 0)
    if x_values.size < 100:
        raise ValueError(f"Trace row is too short: only {x_values.size} image columns detected.")
    segments = np.split(x_values, np.flatnonzero(np.diff(x_values) > 1) + 1)
    x_values = max(segments, key=len)
    if x_values.size < 100:
        raise ValueError(f"Continuous trace segment is too short: only {x_values.size} image columns detected.")

    y_values = np.full(x_values.size, np.nan, dtype=np.float32)
    for index, x_value in enumerate(x_values):
        y_pixels = np.flatnonzero(row_mask[:, x_value]) + y0
        if y_pixels.size:
            y_values[index] = float(np.median(y_pixels))

    valid = np.isfinite(y_values)
    coverage = float(valid.mean())
    if coverage < 0.9:
        raise ValueError(f"Trace coverage is too low ({coverage:.1%}); mark this photo unreadable.")

    y_values = np.interp(np.arange(y_values.size), np.flatnonzero(valid), y_values[valid]).astype(np.float32)
    local_median = median_filter(y_values, size=SPIKE_MEDIAN_WINDOW, mode="nearest")
    spike_candidates = np.abs(y_values - local_median) > SPIKE_DEVIATION_THRESHOLD_PX
    candidate_indices = np.flatnonzero(spike_candidates)
    spike_runs = np.split(candidate_indices, np.flatnonzero(np.diff(candidate_indices) > 1) + 1)
    corrected = np.zeros(y_values.size, dtype=bool)
    for run in spike_runs:
        if run.size == 0 or run.size > MAX_ISOLATED_SPIKE_WIDTH:
            continue
        start = int(run[0])
        end = int(run[-1]) + 1
        if start == 0 or end == y_values.size:
            continue
        y_values[start:end] = np.interp(
            np.arange(start, end),
            (start - 1, end),
            (y_values[start - 1], y_values[end]),
        )
        corrected[start:end] = True

    baseline = float(np.median(y_values))
    return x_values.astype(np.float32), baseline - y_values, valid, int(corrected.sum())


def compare_with_reference(
    extracted: dict[str, tuple[np.ndarray, np.ndarray]],
    reference_path: Path,
    max_lag_samples: int,
    seconds_per_pixel: float | None,
    millivolts_per_pixel: float | None,
) -> dict[str, object]:
    with np.load(reference_path) as source:
        reference_signal = source["signal"].astype(np.float32)
        reference_leads = [str(name) for name in source["lead_names"]]
        sampling_rate = float(source["sampling_rate"]) if "sampling_rate" in source.files else None

    results = {}
    for lead, (x_values, pixels_from_baseline) in extracted.items():
        if lead not in reference_leads:
            raise ValueError(f"Reference signal is missing lead {lead}.")

        reference = reference_signal[:, reference_leads.index(lead)]
        reference = reference - np.median(reference)
        if seconds_per_pixel is not None and millivolts_per_pixel is not None:
            if sampling_rate is None:
                raise ValueError("Reference signal has no sampling_rate; calibrated comparison cannot be computed.")
            time_seconds = (x_values - x_values.min()) * seconds_per_pixel
            reference_times = np.arange(len(reference)) / sampling_rate
            comparison_end = min(float(time_seconds[-1]), float(reference_times[-1]))
            comparison_times = np.arange(
                int(np.floor(comparison_end * sampling_rate)) + 1,
                dtype=np.float64,
            ) / sampling_rate
            photo_values = np.interp(comparison_times, time_seconds, pixels_from_baseline)
            reference = np.interp(comparison_times, reference_times, reference)
        else:
            photo_values = np.interp(
                np.linspace(float(x_values.min()), float(x_values.max()), len(reference)),
                x_values,
                pixels_from_baseline,
            )
        photo_values = (photo_values - photo_values.mean()) / (photo_values.std() + 1e-8)
        reference = reference / (reference.std() + 1e-8)

        correlations = correlate(photo_values, reference, mode="full", method="fft") / len(reference)
        lags = correlation_lags(len(photo_values), len(reference), mode="full")
        within_limit = np.abs(lags) <= max_lag_samples
        best_index = int(np.argmax(correlations[within_limit]))
        results[lead] = {
            "zero_lag_correlation": float(np.corrcoef(photo_values, reference)[0, 1]),
            "best_correlation_with_alignment": float(correlations[within_limit][best_index]),
            "alignment_lag_samples": int(lags[within_limit][best_index]),
            "note": "Normalized shape correlation only; not voltage/time calibration or diagnostic accuracy.",
        }
        if seconds_per_pixel is not None and millivolts_per_pixel is not None:
            if sampling_rate is None:
                raise ValueError("Reference NPZ has no sampling_rate; calibrated error cannot be computed.")
            time_seconds = (x_values - x_values.min()) * seconds_per_pixel
            measured_mv = pixels_from_baseline * millivolts_per_pixel
            reference_mv = reference_signal[:, reference_leads.index(lead)]
            reference_mv = reference_mv - np.median(reference_mv)
            reference_times = np.arange(len(reference_mv)) / sampling_rate
            reference_at_photo_times = np.interp(time_seconds, reference_times, reference_mv)
            results[lead]["calibrated_rmse_mV"] = float(
                np.sqrt(np.mean((measured_mv - reference_at_photo_times) ** 2))
            )
            results[lead]["calibrated_time_end_seconds"] = float(time_seconds[-1])

    return {
        "reference_file": str(reference_path),
        "max_alignment_lag_samples": max_lag_samples,
        "leads": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract one pixel-coordinate trace from a single-lead or standard three-row monitor photo."
    )
    parser.add_argument("image", type=Path, help="JPEG or PNG monitor photo with one trace or I/II/III rows")
    parser.add_argument(
        "--single-lead-name",
        choices=("I", "II", "III"),
        default="II",
        help="Confirm the lead name when the image contains only one trace row.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("ml/data/digitized_monitor_photo"))
    parser.add_argument(
        "--plot-corners",
        type=parse_plot_corners,
        help="Optional grid corners as TL;TR;BR;BL x,y pairs, for example '78,255;3738,286;3740,1410;65,1278'.",
    )
    parser.add_argument(
        "--reference-image",
        type=Path,
        help="Optional clean rendering of the photographed screen; use feature matching to align without downsampling.",
    )
    parser.add_argument("--reference-npz", type=Path, help="Optional paired NPZ with signal and lead_names arrays")
    parser.add_argument("--max-lag-samples", type=int, default=750)
    parser.add_argument(
        "--seconds-per-pixel",
        type=float,
        help="Calibrated time scale; with --reference-image, specify this in reference-image pixels.",
    )
    parser.add_argument(
        "--millivolts-per-pixel",
        type=float,
        help="Calibrated voltage scale; with --reference-image, specify this in reference-image pixels.",
    )
    args = parser.parse_args()
    if args.plot_corners is not None and args.reference_image is not None:
        parser.error("--plot-corners and --reference-image are mutually exclusive.")
    if (args.seconds_per_pixel is None) != (args.millivolts_per_pixel is None):
        parser.error("--seconds-per-pixel and --millivolts-per-pixel must be provided together.")
    if args.seconds_per_pixel is not None and (args.seconds_per_pixel <= 0 or args.millivolts_per_pixel <= 0):
        parser.error("Calibration scales must be positive.")

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit("Could not read image. Convert HEIC to JPEG or PNG first.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.reference_image is not None:
        reference = cv2.imread(str(args.reference_image), cv2.IMREAD_COLOR)
        if reference is None:
            raise SystemExit("Could not read --reference-image.")
        image, registration_scale, inlier_count, median_error = register_to_reference(image, reference)
        rectified_path = args.out_dir / "reference_rectified.jpg"
        if not cv2.imwrite(str(rectified_path), image):
            raise OSError(f"Could not write registered image: {rectified_path}")
        if args.seconds_per_pixel is not None:
            args.seconds_per_pixel /= registration_scale
            args.millivolts_per_pixel /= registration_scale
            print(
                f"Reference calibration converted to {args.seconds_per_pixel:.8g} s/px and "
                f"{args.millivolts_per_pixel:.8g} mV/px."
            )
        print(
            f"Reference-aligned image saved to {rectified_path} ({image.shape[1]}x{image.shape[0]}), "
            f"scale={registration_scale:.3f}x, inliers={inlier_count}, median error={median_error:.2f}px."
        )
    elif args.plot_corners is not None:
        image = rectify_plot(image, args.plot_corners)
        rectified_path = args.out_dir / "rectified_plot.jpg"
        if not cv2.imwrite(str(rectified_path), image):
            raise OSError(f"Could not write rectified plot: {rectified_path}")
        print(f"Perspective-corrected plot saved to {rectified_path} ({image.shape[1]}x{image.shape[0]}).")

    mask = make_trace_mask(image)
    bands = find_trace_bands(mask)
    if len(bands) == 3:
        bands = [bands[1]]
        selected_lead = "II"
        print("Three ordered I/II/III rows detected; extracting only the middle Lead II row.")
    else:
        selected_lead = args.single_lead_name
        print(f"One trace row detected; treating it as Lead {selected_lead} as specified.")
    csv_path = args.out_dir / "traces_px.csv"
    overlay = image.copy()
    rows = []
    extracted = {}

    for lead_name, (y0, y1) in zip((selected_lead,), bands):
        x_values, y_from_baseline, valid, corrected_spikes = extract_trace(mask, y0, y1)
        extracted[lead_name] = (x_values, y_from_baseline)
        y_baseline = float(np.median(np.flatnonzero(mask[y0:y1]).reshape(-1) % (y1 - y0)) + y0)
        for point_index, (x_value, y_value) in enumerate(zip(x_values, y_from_baseline)):
            point = {
                "lead_row": lead_name,
                "point": point_index,
                "x_pixel": round(float(x_value), 2),
                "y_from_baseline_pixel": round(float(y_value), 2),
            }
            if args.seconds_per_pixel is not None and args.millivolts_per_pixel is not None:
                point["time_seconds"] = round(float((x_value - x_values.min()) * args.seconds_per_pixel), 6)
                point["voltage_mV"] = round(float(y_value * args.millivolts_per_pixel), 6)
            rows.append(point)
        points = np.column_stack((x_values, y_baseline - y_from_baseline)).round().astype(np.int32)
        cv2.polylines(overlay, [points], False, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.rectangle(overlay, (0, y0), (image.shape[1] - 1, y1), (255, 0, 0), 2)
        print(
            f"Lead {lead_name}: {len(x_values)} points, green-pixel coverage {valid.mean():.1%}, "
            f"isolated spikes corrected {corrected_spikes}"
        )

    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    overlay_path = args.out_dir / "trace_overlay.jpg"
    if not cv2.imwrite(str(overlay_path), overlay):
        raise OSError(f"Could not write overlay: {overlay_path}")
    print(f"Pixel-coordinate traces saved to {csv_path}")
    print(f"Review overlay saved to {overlay_path}")
    if args.seconds_per_pixel is None:
        print(f"Lead {selected_lead} values are in pixels; explicit calibration is required for seconds and mV.")
    else:
        print(f"Lead {selected_lead} calibrated at {args.seconds_per_pixel:g} s/px and {args.millivolts_per_pixel:g} mV/px.")

    if args.reference_npz:
        comparison = compare_with_reference(
            extracted,
            args.reference_npz,
            args.max_lag_samples,
            args.seconds_per_pixel,
            args.millivolts_per_pixel,
        )
        metrics_path = args.out_dir / "reference_comparison.json"
        metrics_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        for lead, metrics in comparison["leads"].items():
            print(
                f"Lead {lead}: zero-lag corr={metrics['zero_lag_correlation']:.3f}, "
                f"aligned corr={metrics['best_correlation_with_alignment']:.3f}, "
                f"lag={metrics['alignment_lag_samples']} samples"
            )
        print(f"Reference comparison saved to {metrics_path}")


if __name__ == "__main__":
    main()
