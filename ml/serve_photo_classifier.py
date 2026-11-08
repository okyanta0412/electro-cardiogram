from __future__ import annotations

import argparse
import base64
import binascii
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import joblib
import numpy as np

from monitor_photo_classifier import MODEL_PATH, extract_features_from_image


MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_REQUEST_BYTES = 14 * 1024 * 1024
PAC_MODEL_PATH = Path(__file__).resolve().parent / "models" / "pac_detector.joblib"
WARNING = (
    "研究用試作です。PAC候補は数値信号のみで評価した別検出器の出力で、"
    "写真上の精度は未検証です。医療診断や治療判断には使用できません。"
)


def make_handler(classifier, pac_detector):
    class PhotoClassifierHandler(BaseHTTPRequestHandler):
        def send_json(self, status: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:
            self.send_json(204, {})

        def do_GET(self) -> None:
            if self.path != "/api/health":
                self.send_json(404, {"error": "Not found"})
                return
            self.send_json(
                200,
                {"ready": True, "labels": [*classifier.classes_.tolist(), "PAC"], "warning": WARNING},
            )

        def do_POST(self) -> None:
            if self.path != "/api/classify":
                self.send_json(404, {"error": "Not found"})
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
                image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    self.send_json(422, {"error": "Could not decode the uploaded image."})
                    return

                features = extract_features_from_image(image).reshape(1, -1)
                pac_score = float(pac_detector["classifier"].decision_function(features)[0])
                pac_candidate = pac_score >= float(pac_detector["decision_threshold"])
                predicted_label = "PAC" if pac_candidate else str(classifier.predict(features)[0])
                self.send_json(
                    200,
                    {
                        "label": predicted_label,
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
    parser = argparse.ArgumentParser(description="Serve the local ECG photo-classification demo API.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address; keep this on loopback for local demo use.")
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
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(artifact["classifier"], pac_detector),
    )
    print(f"ECG photo demo API ready at http://{args.host}:{args.port}/api/health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping ECG photo demo API.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()