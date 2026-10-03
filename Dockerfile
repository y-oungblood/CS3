# Movie Swipe web app. Only the app runtime: no LensKit, raw data or simulation code.
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FLET_FORCE_WEB_SERVER=1 \
    HOST=0.0.0.0 \
    PORT=10000 \
    STORAGE_BACKEND=firestore

WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY recsys/ recsys/
COPY app/ app/
COPY data/artifacts/deploy/ data/artifacts/deploy/

RUN useradd --create-home appuser && chown -R appuser /srv
USER appuser

EXPOSE 10000
# Firestore credentials come from the FIREBASE_CREDENTIALS secret set on the host.
CMD ["python", "app/main.py", "--web"]
