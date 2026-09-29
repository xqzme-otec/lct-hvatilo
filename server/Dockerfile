FROM python:3.13-slim-trixie AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir  --target=/app/deps -r server/requirements.txt


### Runtime
FROM gcr.io/distroless/python3-debian13:nonroot
WORKDIR /app

COPY --from=builder /app/deps /app/deps
ENV PYTHONPATH="/app/deps:/app"
ENV PYTHONUNBUFFERED=1
COPY . ./server
USER nonroot
EXPOSE 8000
CMD ["server/entrypoint.py"]