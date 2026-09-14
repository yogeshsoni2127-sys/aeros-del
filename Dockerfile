FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ML inference stack (CSV-trained XGB/LGBM + TFT) installs here so the
# deployed service serves real model predictions, not baseline.
# Local minimal installs can still use requirements-core.txt alone
# (the app degrades gracefully without the ML wheels).
COPY backend/requirements-core.txt backend/requirements-ml.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-core.txt -r /tmp/requirements-ml.txt

COPY . .

EXPOSE 8000

CMD ["python", "backend/run.py"]