FROM python:3.11-slim

# libgl1/libglib — нужны opencv-python-headless и paddleocr для работы с изображениями
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline/ pipeline/
COPY scripts/ scripts/

# Чекпоинт RF-DETR скачивается лениво при первом инференсе (huggingface_hub,
# см. pipeline/rfdetr_infer.py) и кэшируется в /root/.cache/huggingface —
# смонтируй volume на этот путь, чтобы не скачивать заново при каждом restart:
#   docker run -v hf_cache:/root/.cache/huggingface ...

ENTRYPOINT ["python", "scripts/run_pipeline.py"]
CMD ["--help"]
