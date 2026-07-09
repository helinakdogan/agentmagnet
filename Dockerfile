FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY sdk/ /app/sdk/
COPY migrations/ /app/migrations/

RUN pip install --no-cache-dir "/app/sdk[postgres]"

ENV MAGNET_MIGRATIONS_DIR=/app/migrations \
    TOKENIZERS_PARALLELISM=false

EXPOSE 8000

CMD ["agent-magnet-http"]
