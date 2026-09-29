FROM python:3.13-slim-trixie AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir  --target=/app/deps -r requirements.txt
RUN mkdir -p /app/models && \
    curl -L -o /app/models/vertebrae_with_ribs.pt https://github.com/xqzme-otec/lct-hvatilo/releases/download/weights-v1/vertebrae_with_ribs.pt

RUN apt-get update && apt-get install -y --no-install-recommends libxcb1

### Runtime
FROM gcr.io/distroless/python3-debian13:nonroot
WORKDIR /app
COPY --from=builder /usr/lib/x86_64-linux-gnu/libxcb.so.1 /usr/lib/x86_64-linux-gnu/
COPY --from=builder /app/deps /app/deps
COPY --from=builder /app/models /app/models
ENV PYTHONPATH="/app/deps:/app"
ENV PYTHONUNBUFFERED=1
COPY . .
USER nonroot
EXPOSE 8000
CMD ["server/entrypoint.py"]