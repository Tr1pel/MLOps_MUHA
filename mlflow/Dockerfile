# syntax=docker/dockerfile:1
FROM python:3.12.12-slim-bookworm
ARG MLFLOW_VERSION=3.18.0
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
# psycopg2-binary: PostgreSQL backend store driver; boto3: S3 client used by the server to reach MinIO.
RUN pip install \
        "mlflow==${MLFLOW_VERSION}" \
        "psycopg2-binary==2.9.10" \
        "boto3==1.40.61" \
    && groupadd --gid 10002 mlflow \
    && useradd --uid 10002 --gid mlflow --no-create-home mlflow
USER 10002:10002
EXPOSE 5000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3)"
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
