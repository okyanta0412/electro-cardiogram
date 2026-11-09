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
WARNING = (
    "研究用試作です。615件の数値信号による患者単位評価で均衡正解率は0.723、20.7%を判定保留にしました。"
    "写真精度は未検証です。PAC候補も別検出器の出力です。医療診断や治療判断には使用できません。"
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


def make_handler(classifier, pac_detector, pvc_af_review_margin: float = 2.0):
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
                    {"ready": True, "labels": [*classifier.classes_.tolist(), "PAC"], "warning": WARNING},
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

                waveform = extract_waveform_from_image(image)
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
                self.send_json(
                    200,
                    {
                        "label": predicted_label,
                        "review_reason": review_reason,
                        "pvc_af_margin": pvc_af_margin,
                        "pac_candidate": pac_candidate,
                        "pac_score": pac_score,
                        "pac_threshold": pac_detector["decision_threshold"],
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
    args = parser.parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}. Run ml/monitor_photo_classifier.py first.")
    if not args.pac_model.exists():
        raise SystemExit(f"PAC model not found: {args.pac_model}. Run ml/train_pac_detector.py first.")

    artifact = joblib.load(args.model)
    pac_detector = joblib.load(args.pac_model)
    port = int(os.environ.get("PORT", args.port))
    server = ThreadingHTTPServer(
        (args.host, port),
        make_handler(
            artifact["classifier"],
            pac_detector,
            float(artifact.get("pvc_af_review_margin", 2.0)),
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