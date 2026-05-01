# Minimal MLflow tracking server image with proxied artifact serving.
#
# Backend store:        sqlite (file under /mlflow/mlflow.db).
# Artifact destination: /mlflow/artifacts inside the container.
# Artifact serving:     HTTP proxy via the tracking server itself
#                       (``--serve-artifacts``) — clients upload through
#                       the API instead of writing directly to the
#                       artifact path. Without this, ``mlflow.log_artifact``
#                       tries to ``mkdir /mlflow/artifacts`` on the host,
#                       which fails on systems where ``/mlflow`` is not
#                       writable.
#
# The /mlflow directory is bind-mounted from the host as ``mlflow-data/``
# so the sqlite DB and artifact tree survive container restarts and can
# be inspected from the host.
FROM python:3.11-slim

RUN pip install --no-cache-dir \
        mlflow==2.14.1 \
        sqlalchemy \
    && mkdir -p /mlflow/artifacts

WORKDIR /mlflow

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", \
     "--serve-artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
