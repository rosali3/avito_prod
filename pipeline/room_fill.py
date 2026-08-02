"""
Пост-процессинг для UNet: Canny+dilate+connectedComponents+majority vote —
лучший результат по room F1 из всех опробованных методов на UGC test
(room F1 0.568 -> 0.588, см. claude_instseg_compare/docs/
room_postprocessing_experiments.md). Границы комнат ищутся классическим
CV прямо на исходном фото (линии стен на печатном плане надёжнее видны
Canny-эджами, чем через дырявую предсказанную wall-маску), затем каждая
найденная замкнутая область заливается доминирующим классом модели внутри
неё.

Портировано без изменений из claude_instseg_compare/eval/
image_based_room_regions.py (только room-fill часть, без CLAHE/extent-crop/
adaptiveThreshold — те эксперименты не превзошли эту базовую версию).
"""
from __future__ import annotations

import cv2
import numpy as np

ROOM_TYPE_IDS = [1, 2, 3, 4, 5]  # living, bedroom, bathroom, kitchen, balcony

DEFAULT_CANNY_LO = 80
DEFAULT_CANNY_HI = 200
DEFAULT_DILATE_ITERS = 2
DEFAULT_MIN_AREA_FRAC = 0.01
DEFAULT_MIN_ROOM_FRAC = 0.3


def detect_room_regions(image_bgr: np.ndarray, canny_lo: int = DEFAULT_CANNY_LO,
                         canny_hi: int = DEFAULT_CANNY_HI, dilate_iters: int = DEFAULT_DILATE_ITERS,
                         min_area_frac: float = DEFAULT_MIN_AREA_FRAC) -> np.ndarray:
    """Возвращает HxW int32: 0 = граница/шум/слишком мелкая область, 1..N = room-region id."""
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, canny_lo, canny_hi)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=dilate_iters)

    interior = (edges == 0).astype(np.uint8)
    n, labels = cv2.connectedComponents(interior, connectivity=8)

    min_area = min_area_frac * h * w
    out = np.zeros((h, w), dtype=np.int32)
    next_id = 1
    for comp_id in range(1, n):
        comp_mask = labels == comp_id
        if comp_mask.sum() < min_area:
            continue
        out[comp_mask] = next_id
        next_id += 1
    return out


def majority_fill_by_regions(pred_label_map: np.ndarray, region_map: np.ndarray,
                              min_room_frac: float = DEFAULT_MIN_ROOM_FRAC) -> np.ndarray:
    out = pred_label_map.copy()
    for region_id in range(1, region_map.max() + 1):
        m = region_map == region_id
        vals = pred_label_map[m]
        room_vals = vals[np.isin(vals, ROOM_TYPE_IDS)]
        if len(room_vals) == 0:
            continue
        counts = np.bincount(room_vals, minlength=8)
        majority = int(counts.argmax())
        if counts[majority] / len(vals) >= min_room_frac:
            out[m] = majority
    return out


def apply_room_fill(image_bgr: np.ndarray, semantic_mask: np.ndarray) -> np.ndarray:
    """Единая точка входа: сырая semantic-маска -> заполненная по границам
    с исходного фото."""
    regions = detect_room_regions(image_bgr)
    return majority_fill_by_regions(semantic_mask, regions)
