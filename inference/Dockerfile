FROM python:3.13-slim

WORKDIR /app

# Устанавливаем системные зависимости для OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY inference/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/models && \
    curl -L -o /app/models/vertebrae_with_ribs.pt https://github.com/xqzme-otec/lct-hvatilo/releases/download/weights-v1/vertebrae_with_ribs.pt
RUN mkdir -p /app/models && \
    curl -L -o /app/models/art_paste_strong.pt https://github.com/xqzme-otec/lct-hvatilo/releases/download/weights-v1/art_paste_strong.pt

COPY inference/ /app/inference/
COPY inference/requirements.txt /app/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
