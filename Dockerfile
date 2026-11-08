FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY ml/requirements.txt /app/ml/requirements.txt
RUN pip install -r /app/ml/requirements.txt

COPY index.html script.js style.css /app/
COPY ml/digitize_monitor_photo.py ml/monitor_photo_classifier.py ml/serve_photo_classifier.py /app/ml/
COPY ml/models/monitor_photo_classifier.joblib ml/models/pac_detector.joblib /app/ml/models/

EXPOSE 10000

CMD ["python", "ml/serve_photo_classifier.py", "--host", "0.0.0.0"]