FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=America/New_York

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 tzdata curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

RUN mkdir -p /app/models /app/state /app/logs && \
    useradd -m -u 1000 bot && chown -R bot:bot /app
USER bot

HEALTHCHECK --interval=60s --timeout=10s --start-period=90s --retries=3 \
  CMD python -c "import json,os,sys,datetime as dt; \
p='/app/state/heartbeat.json'; \
sys.exit(1) if not os.path.exists(p) else None; \
age=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(json.load(open(p))['ts'])).total_seconds(); \
sys.exit(0 if age < 300 else 1)"

CMD ["python", "-m", "app.run_bot"]
