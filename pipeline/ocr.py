"""
OCR извлечение подписей площади комнат с плана.

Движок: PaddleOCR **без какой-либо предобработки** — по замерам на
ручной разметке (67 подписей площади, claude_instseg_compare/docs/
furniture_experiments_log.md) это безоговорочный лидер (38.8% recall),
почти вдвое лучше EasyOCR+CLAHE (20.9%) и заметно лучше самого PaddleOCR
+ CLAHE (14.9% — предобработка ему СПЕЦИФИЧНО вредит, поэтому она здесь
намеренно не применяется, в отличие от preprocessing.py для сегментации).

Recall даже у лучшего варианта низкий (обычно находится в лучшем случае
половина подписей на плане) — это ЗАВЕДОМО слабый/опциональный сигнал,
не источник истины. Дальнейшая логика (какой сборщик использует OCR)
должна уметь работать и без него.
"""
from __future__ import annotations

import re

import numpy as np

DECIMAL_RE = re.compile(r"^\d{1,2}[.,]\d{1,2}$")

_ocr_engine = None


def _get_engine():
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR
        _ocr_engine = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
    return _ocr_engine


def extract_area_labels(image_bgr: np.ndarray) -> list[dict]:
    """Возвращает список кандидатов в подписи площади:
    [{"bbox": [[x,y],...] (4 точки), "text": str, "value_m2": float}, ...]
    Без предобработки, без geometric-привязки к конкретной комнате
    (это уже отдельная эвристика поверх, см. furniture/data_prep/
    room_area_ocr.py в основном репозитории, если понадобится)."""
    engine = _get_engine()
    result = engine.ocr(image_bgr, cls=False)
    if not result or result[0] is None:
        return []

    candidates = []
    for bbox, (text, conf) in result[0]:
        text_norm = text.strip().replace(",", ".")
        if DECIMAL_RE.match(text.strip()):
            try:
                value = float(text_norm)
            except ValueError:
                continue
            candidates.append({"bbox": bbox, "text": text.strip(), "value_m2": value, "conf": float(conf)})
    return candidates
