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

INT_RE = re.compile(r"^\d{1,2}$")
DECIMAL_RE = re.compile(r"^\d{1,2}[.,]\d{1,2}$")
HEIGHT_RE = re.compile(r"^h\s*=", re.IGNORECASE)

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


def _boxes_from_result(res) -> list[dict]:
    texts = res.get("rec_texts", []) if hasattr(res, "get") else res["rec_texts"]
    scores = res.get("rec_scores", []) if hasattr(res, "get") else res["rec_scores"]
    polys = res.get("rec_polys", None) if hasattr(res, "get") else res.get("rec_polys")
    boxes = []
    for i, text in enumerate(texts):
        poly = polys[i] if polys is not None else None
        if poly is not None:
            xs, ys = poly[:, 0], poly[:, 1]
            x0, y0, x1, y1 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
        else:
            x0 = y0 = x1 = y1 = 0.0
        boxes.append({
            "text": text.strip(), "conf": float(scores[i]),
            "bbox": poly.tolist() if poly is not None else None,
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "cx": (x0 + x1) / 2, "h": y1 - y0,
        })
    return boxes


def _find_area_candidates(boxes: list[dict]) -> set[int]:
    """Та же эвристика, что furniture/data_prep/ocr_visualize.py в
    research-репозитории: на плане в центре комнаты обычно печатают ДВА
    числа друг под другом (сверху — номер комнаты, маленькое целое;
    снизу — площадь, десятичное число). Без этой привязки OCR находит
    ЛЮБОЕ десятичное число на плане, включая размеры стен/проёмов
    (напр. "2.35" на размерной линии) — их не отличить от площади по
    одному только формату текста."""
    area_idx: set[int] = set()
    for i, a in enumerate(boxes):
        if not INT_RE.match(a["text"]):
            continue
        for j, b in enumerate(boxes):
            if i == j:
                continue
            b_text = b["text"].replace(",", ".")
            if HEIGHT_RE.match(b_text) or not DECIMAL_RE.match(b_text):
                continue
            # b должен быть прямо под a (широкий допуск — реальные фото
            # часто повёрнуты/смещены, не жёсткий порог по пикселям)
            avg_h = (a["h"] + b["h"]) / 2 or 1.0
            gap = b["y0"] - a["y1"]
            if -avg_h <= gap <= avg_h * 3 and abs(a["cx"] - b["cx"]) < avg_h * 4:
                area_idx.add(j)
    return area_idx


def extract_area_labels(image_bgr: np.ndarray) -> list[dict]:
    """Возвращает список кандидатов в подписи площади — ТОЛЬКО десятичные
    числа, над которыми стоит маленькое целое (номер комнаты), — а не
    любое десятичное число на плане (иначе в кандидаты попадают и размеры
    стен на размерных линиях):
    [{"bbox": [[x,y],...] (4 точки) или None, "text": str,
      "value_m2": float, "conf": float}, ...]
    Без предобработки (см. модуль docstring). Без дальнейшей geometric-
    привязки к конкретной комнате — это уже отдельный шаг поверх, см.
    furniture/data_prep/room_area_ocr.py в research-репозитории."""
    engine = _get_engine()
    results = engine.predict(image_bgr)
    if not results:
        return []

    candidates = []
    for res in results:
        boxes = _boxes_from_result(res)
        area_idx = _find_area_candidates(boxes)
        for j in area_idx:
            b = boxes[j]
            try:
                value = float(b["text"].replace(",", "."))
            except ValueError:
                continue
            candidates.append({
                "bbox": b["bbox"], "text": b["text"], "value_m2": value, "conf": b["conf"],
            })
    return candidates
