# Contest harness image. Pin base + deps; nothing is downloaded at runtime.
# openjiuwen requires Python >=3.11,<3.14 — 3.13 is the newest compatible.
FROM python:3.13.18-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp \
    PYTHONPATH=/app/src

WORKDIR /app

# System packages needed for local tool execution inside tasks.
RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY run_agent.py /app/run_agent.py
COPY src/harness /app/src/harness
COPY scripts /app/scripts
# Ranking weights → /app/models (rank.py falls back here inside the image)
COPY artifacts/models /app/models

# Default entry matches jury: python /app/run_agent.py --tasks-dir /data/tasks
ENTRYPOINT ["python", "/app/run_agent.py"]
CMD ["--tasks-dir", "/data/tasks"]
