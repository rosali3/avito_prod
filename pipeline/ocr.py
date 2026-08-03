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
        # API PaddleOCR 3.x (PP-OCRv5 unified pipeline) — старые kwargs
        # use_angle_cls/show_log больше не существуют. Отключаем doc-шаги,
        # которые нам не нужны для плоских сканов/фото плана (без разворота
        # страницы книгой и т.п.) — быстрее и ближе к старому поведению
        # use_angle_cls=False.
        from paddleocr import PaddleOCR
        _ocr_engine = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            # enable_mkldnn=False обязателен: с MKL-DNN/oneDNN backend'ом
            # (дефолт для CPU) падает с
            # "NotImplementedError: ConvertPirAttribute2RuntimeAttribute
            # not support ... onednn_instruction.cc" — известный баг новой
            # PIR-исполнительной системы paddlepaddle 3.x с oneDNN на CPU.
            enable_mkldnn=False,
        )
    return _ocr_engine


def extract_area_labels(image_bgr: np.ndarray) -> list[dict]:
    """Возвращает список кандидатов в подписи площади:
    [{"bbox": [[x,y],...] (4 точки, rec_polys) или None, "text": str,
      "value_m2": float, "conf": float}, ...]
    Без предобработки, без geometric-привязки к конкретной комнате
    (это уже отдельная эвристика поверх, см. furniture/data_prep/
    room_area_ocr.py в основном репозитории, если понадобится)."""
    engine = _get_engine()
    results = engine.predict(image_bgr)
    if not results:
        return []

    candidates = []
    for res in results:
        texts = res.get("rec_texts", []) if hasattr(res, "get") else res["rec_texts"]
        scores = res.get("rec_scores", []) if hasattr(res, "get") else res["rec_scores"]
        polys = res.get("rec_polys", None) if hasattr(res, "get") else res.get("rec_polys")
        for i, text in enumerate(texts):
            text_stripped = text.strip()
            if not DECIMAL_RE.match(text_stripped):
                continue
            try:
                value = float(text_stripped.replace(",", "."))
            except ValueError:
                continue
            bbox = polys[i].tolist() if polys is not None else None
            candidates.append({
                "bbox": bbox,
                "text": text_stripped,
                "value_m2": value,
                "conf": float(scores[i]),
            })
    return candidates
