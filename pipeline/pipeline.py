"""
Единая точка входа: фото плана -> результат (сегментация + мебель-заглушка + OCR).

    предобработка (CLAHE) -> RF-DETR -> [мебель: заглушка] -> OCR -> combine

Мебельный шаг закомментирован по существу (см. furniture_stub.py) — вызов
оставлен явным, чтобы было видно, где он подключится, когда модель будет
готова к использованию на реальных фото.
"""
from __future__ import annotations

import numpy as np

from .preprocessing import preprocess_image
from .rfdetr_infer import run_rfdetr, assemble_semantic_mask, DEFAULT_THRESHOLD, DEFAULT_NMS_IOU
from .unet_infer import run_unet
from .room_fill import apply_room_fill
from .furniture_stub import run_furniture_detector
from .ocr import extract_area_labels


def run_pipeline(image_bgr: np.ndarray, model: str = "rfdetr", apply_clahe: bool = True,
                  rfdetr_threshold: float = DEFAULT_THRESHOLD,
                  rfdetr_nms_iou: float = DEFAULT_NMS_IOU,
                  unet_room_fill: bool = True,
                  run_ocr: bool = True) -> dict:
    """model: "rfdetr" (рекомендуется, лучше на UGC test) или "unet" (+ Canny
    room-fill постобработка, включена по умолчанию — единственный сценарий,
    где этот пост-процессинг реально помогает, см. room_fill.py).

    Возвращает:
    {
      "instances": [...],           # инстансы RF-DETR после mask-NMS (только model="rfdetr", иначе [])
      "semantic_mask": np.ndarray,  # [H,W] argmax-карта, id 1-7 (0=фон)
      "furniture": [],              # ЗАГЛУШКА — всегда пусто, см. furniture_stub.py
      "area_labels": [...],         # OCR-кандидаты в подпись площади (слабый сигнал)
    }
    """
    if model not in ("rfdetr", "unet"):
        raise ValueError(f"model должен быть 'rfdetr' или 'unet', получено: {model!r}")

    h, w = image_bgr.shape[:2]
    image_for_model = preprocess_image(image_bgr, apply_clahe_=apply_clahe)

    if model == "rfdetr":
        instances = run_rfdetr(image_for_model, threshold=rfdetr_threshold, nms_iou=rfdetr_nms_iou)
        semantic_mask = assemble_semantic_mask(instances, h, w)
    else:
        instances = []
        semantic_mask = run_unet(image_for_model)
        if unet_room_fill:
            # Canny-fill применяется к НЕобработанному CLAHE фото — граница
            # ищется по исходным пикселям, см. эксперимент в room_fill.py
            semantic_mask = apply_room_fill(image_bgr, semantic_mask)

    # --- Мебель: реальный вызов пока закомментирован, см. furniture_stub.py ---
    # furniture_instances = run_furniture_detector(image_bgr)  # TODO: не готово для UGC
    furniture_instances = run_furniture_detector(image_bgr)  # заглушка -> всегда []

    area_labels = extract_area_labels(image_bgr) if run_ocr else []

    return {
        "instances": instances,
        "semantic_mask": semantic_mask,
        "furniture": furniture_instances,
        "area_labels": area_labels,
    }
