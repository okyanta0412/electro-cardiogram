from __future__ import annotations

import argparse
import base64
import binascii
import json
import mimetypes
import os
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlsplit

import cv2
import joblib
import numpy as np
import pillow_heif
from PIL import Image, ImageOps

from monitor_photo_classifier import (
    MODEL_PATH,
    extract_features_from_waveform,
    extract_waveform_from_image,
)
from train_labeled_photo_prototype import extract_labeled_photo_waveform, predict_labeled_photo

pillow_heif.register_heif_opener()


MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_REQUEST_BYTES = 14 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
RATE_LIMIT_REQUESTS = 20
RATE_LIMIT_WINDOW_SECONDS = 60
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_FILES = {
    "/": PROJECT_ROOT / "index.html",
    "/index.html": PROJECT_ROOT / "index.html",
    "/style.css": PROJECT_ROOT / "style.css",
    "/script.js": PROJECT_ROOT / "script.js",
}
PAC_MODEL_PATH = Path(__file__).resolve().parent / "models" / "pac_detector.joblib"
ADDITIONAL_MODELS_PATH = Path(__file__).resolve().parent / "models" / "additional_ecg_findings.joblib"
VT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "vt_single_episode_prototype.joblib"
LABELED_PHOTO_MODEL_PATH = Path(__file__).resolve().parent / "models" / "labeled_photo_prototype.joblib"
FINDING_PRIORITY = (
    "ventricular_tachycardia",
    "third_degree_av_block",
    "nonspecific_st_elevation",
    "second_degree_av_block",
    "nonspecific_st_depression",
)
WARNING = (
    "研究用試作です。主分類・追加所見・PAC候補はいずれも写真からの精度が未検証です。"
    "医療診断や治療判断には使用できません。"
)


def score_vt_windows(waveform: np.ndarray, detector: dict[str, object]) -> float:
    signal = np.asarray(waveform, dtype=np.float32).reshape(-1)
    source_window_size = 100
    target_window_size = 360
    step_size = source_window_size // 2
    classifier = detector["classifier"]
    scores = []
    for start in range(0, len(signal) - source_window_size + 1, step_size):
        window = signal[start : start + source_window_size]
        centered = window - float(np.median(window))
        normalized = centered / (float(np.std(centered)) + 1e-6)
        resampled = np.interp(
            np.linspace(0.0, source_window_size - 1, target_window_size),
            np.arange(source_window_size),
            normalized,
        )
        scores.append(float(classifier.decision_function(resampled.reshape(1, -1))[0]))
    return max(scores, default=float("-inf"))


def select_priority_finding(additional_results: dict[str, dict[str, object]]) -> str | None:
    return next(
        (
            name
            for name in FINDING_PRIORITY
            if additional_results.get(name, {}).get("candidate", False)
        ),
        None,
    )


def decode_uploaded_image(image_bytes: bytes) -> np.ndarray:
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is not None:
        return image

    try:
        with Image.open(BytesIO(image_bytes)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image dimensions exceed the 25 megapixel limit.")
            rgb_image = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
        return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("Could not decode image. Use JPEG, PNG, HEIC or HEIF.") from error


def make_handler(
    classifier,
    pac_detector,
    pvc_af_review_margin: float = 2.0,
    additional_detectors: dict[str, object] | None = None,
    vt_detector: dict[str, object] | None = None,
    photo_prototype: dict[str, object] | None = None,
):
    requests_by_client: dict[str, deque[float]] = defaultdict(deque)
    rate_limit_lock = threading.Lock()

    class PhotoClassifierHandler(BaseHTTPRequestHandler):
        def send_json(self, status: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def allow_request(self) -> bool:
            now = time.monotonic()
            client = self.client_address[0]
            with rate_limit_lock:
                recent_requests = requests_by_client[client]
                while recent_requests and now - recent_requests[0] > RATE_LIMIT_WINDOW_SECONDS:
                    recent_requests.popleft()
                if len(recent_requests) >= RATE_LIMIT_REQUESTS:
                    return False
                recent_requests.append(now)
            return True

        def do_OPTIONS(self) -> None:
            self.send_json(204, {})

        def do_GET(self) -> None:
            request_path = unquote(urlsplit(self.path).path)
            if request_path == "/api/health":
                self.send_json(
                    200,
                    {
                        "ready": True,
                        "labels": [*classifier.classes_.tolist(), "REVIEW", "PAC"],
                        "additional_findings": [
                            *(additional_detectors or {}).keys(),
                            *( ["ventricular_tachycardia"] if vt_detector else []),
                        ],
                        "finding_priority": list(FINDING_PRIORITY),
                        "vt_prediction_available": vt_detector is not None,
                        "photo_prototype_available": photo_prototype is not None,
                        "photo_prototype_labels": sorted(set(photo_prototype["labels"].tolist()))
                        if photo_prototype
                        else [],
                        "warning": WARNING,
                    },
                )
                return
            static_path = STATIC_FILES.get(request_path)
            if static_path is None:
                self.send_json(404, {"error": "Not found"})
                return
            body = static_path.read_bytes()
            content_type = mimetypes.guess_type(static_path.name)[0] or "application/octet-stream"
            if content_type.startswith("text/") or content_type == "application/javascript":
                content_type += "; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path != "/api/classify":
                self.send_json(404, {"error": "Not found"})
                return
            if not self.allow_request():
                self.send_json(429, {"error": "Too many requests. Try again in a minute."})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
                    self.send_json(413, {"error": "Request must contain an image no larger than 10 MB."})
                    return
                request = json.loads(self.rfile.read(content_length))
                image_value = request.get("image", "")
                encoded_image = image_value.partition(",")[2] if image_value.startswith("data:") else image_value
                image_bytes = base64.b64decode(encoded_image, validate=True)
                if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
                    self.send_json(413, {"error": "Image must be no larger than 10 MB."})
                    return
                image = decode_uploaded_image(image_bytes)
                if image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
                    self.send_json(413, {"error": "Image dimensions exceed the 25 megapixel limit."})
                    return

                photo_prediction = None
                if photo_prototype is not None:
                    try:
                        photo_prediction = predict_labeled_photo(image, photo_prototype)
                    except ValueError as error:
                        photo_prediction = {"error": str(error)}
                if photo_prediction and photo_prediction.get("label"):
                    self.send_json(
                        200,
                        {
                            "label": photo_prediction["label"],
                            "base_label": None,
                            "priority_finding": None,
                            "photo_prototype_prediction": photo_prediction,
                            "pac_candidate": None,
                            "pac_score": None,
                            "pac_threshold": None,
                            "additional_findings": {},
                            "vt_prediction_available": False,
                            "classification_mode": "labeled_photo_prototype",
                            "warning": (
                                "登録済み教材画像10枚から最も近い波形を返す試作です。"
                                "新しい画像への分類精度は未評価です。"
                            ),
                        },
                    )
                    return

                waveform = (
                    extract_labeled_photo_waveform(image)
                    if photo_prototype is not None
                    else extract_waveform_from_image(image)
                )
                features = extract_features_from_waveform(waveform).reshape(1, -1)
                pac_score = float(
                    pac_detector["classifier"].decision_function(waveform.reshape(1, -1))[0]
                )
                pac_candidate = pac_score >= float(pac_detector["decision_threshold"])
                decision_scores = classifier.decision_function(features)[0]
                class_order = classifier.named_steps["linearsvc"].classes_
                predicted_label = str(class_order[int(np.argmax(decision_scores))])
                review_reason = None
                pvc_af_margin = None
                if predicted_label == "PVC":
                    pvc_score = float(decision_scores[int(np.flatnonzero(class_order == "PVC")[0])])
                    af_score = float(decision_scores[int(np.flatnonzero(class_order == "AF")[0])])
                    pvc_af_margin = pvc_score - af_score
                    if pvc_af_margin < pvc_af_review_margin:
                        predicted_label = "REVIEW"
                        review_reason = "PVCとAFのスコア差が小さいため、判定を保留しました。"
                additional_results = {}
                for name, detector in (additional_detectors or {}).items():
                    score = float(detector.decision_function(waveform.reshape(1, -1))[0])
                    additional_results[name] = {"candidate": score >= 0.0, "score": score}
                if vt_detector is not None:
                    vt_score = score_vt_windows(waveform, vt_detector)
                    additional_results["ventricular_tachycardia"] = {
                        "candidate": vt_score >= 0.0,
                        "score": vt_score,
                    }
                priority_finding = select_priority_finding(additional_results)
                primary_label = (
                    photo_prediction.get("label")
                    if photo_prediction and photo_prediction.get("label")
                    else priority_finding or predicted_label
                )
                self.send_json(
                    200,
                    {
                        "label": primary_label,
                        "base_label": predicted_label,
                        "priority_finding": priority_finding,
                        "photo_prototype_prediction": photo_prediction,
                        "review_reason": review_reason,
                        "pvc_af_margin": pvc_af_margin,
                        "pac_candidate": pac_candidate,
                        "pac_score": pac_score,
                        "pac_threshold": pac_detector["decision_threshold"],
                        "additional_findings": additional_results,
                        "vt_prediction_available": vt_detector is not None,
                        "warning": WARNING,
                    },
                )
            except (ValueError, binascii.Error, json.JSONDecodeError) as error:
                self.send_json(422, {"error": str(error), "warning": WARNING})
            except Exception as error:
                self.send_json(500, {"error": "Classification failed.", "detail": str(error), "warning": WARNING})

        def log_message(self, format: str, *args: object) -> None:
            print(f"{self.address_string()} - {format % args}")

    return PhotoClassifierHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the ECG photo-classification demo and API.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address; use 0.0.0.0 inside a deployment container.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--pac-model", type=Path, default=PAC_MODEL_PATH)
    parser.add_argument("--additional-models", type=Path, default=ADDITIONAL_MODELS_PATH)
    parser.add_argument("--vt-model", type=Path, default=VT_MODEL_PATH)
    parser.add_argument("--photo-prototype", type=Path, default=LABELED_PHOTO_MODEL_PATH)
    args = parser.parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}. Run ml/monitor_photo_classifier.py first.")
    if not args.pac_model.exists():
        raise SystemExit(f"PAC model not found: {args.pac_model}. Run ml/train_pac_detector.py first.")
    if not args.additional_models.exists():
        raise SystemExit(
            f"Additional-finding models not found: {args.additional_models}. "
            "Run ml/train_additional_ecg_findings.py first."
        )
    if not args.vt_model.exists():
        raise SystemExit(f"VT model not found: {args.vt_model}. Run ml/train_vt_single_episode_prototype.py first.")

    artifact = joblib.load(args.model)
    pac_detector = joblib.load(args.pac_model)
    additional_detectors = joblib.load(args.additional_models)
    vt_detector = joblib.load(args.vt_model)
    photo_prototype = joblib.load(args.photo_prototype) if args.photo_prototype.exists() else None
    port = int(os.environ.get("PORT", args.port))
    server = ThreadingHTTPServer(
        (args.host, port),
        make_handler(
            artifact["classifier"],
            pac_detector,
            float(artifact.get("pvc_af_review_margin", 2.0)),
            additional_detectors,
            vt_detector,
            photo_prototype,
        ),
    )
    print(f"ECG photo demo ready at http://{args.host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping ECG photo demo API.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()