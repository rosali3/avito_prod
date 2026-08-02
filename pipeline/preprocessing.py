"""
Предобработка изображения перед инференсом сегментации.

CLAHE-часть — из `clahe_module` (артефакт тиммейта, сравнение методов
предобработки): clip=2.0, tile=8x8 по яркости, для цветного входа
результат реплицируется в 3 канала. Валидация тиммейта (без переобучения
модели): помогает на тусклых/шумных UGC-подобных фото
(macro-mIoU 0.409 -> 0.437), немного вредит на чистых CubiCasa-сканах
(-0.007). Автоопределение "нужен ли CLAHE" не делается — раз этот
пайплайн нацелен на реальные UGC-фото (а не чистую синтетику), CLAHE
включён по умолчанию (`apply_clahe=True`).
"""
from __future__ import annotations

import numpy as np
import cv2

DEFAULT_CLIP = 2.0
DEFAULT_TILE = (8, 8)


def apply_clahe(image: np.ndarray, clip: float = DEFAULT_CLIP, tile=DEFAULT_TILE) -> np.ndarray:
    """CLAHE на изображении. image: grayscale [H,W] или цветное [H,W,3] (BGR).
    Возврат в том же формате."""
    if isinstance(tile, int):
        tile = (tile, tile)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=tuple(tile))
    if image.ndim == 2:
        return clahe.apply(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[2] == 3 else image[..., 0]
    eq = clahe.apply(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)


def preprocess_image(image_bgr: np.ndarray, apply_clahe_: bool = True,
                      clip: float = DEFAULT_CLIP, tile=DEFAULT_TILE) -> np.ndarray:
    """Единая точка входа для предобработки. Сейчас — только CLAHE
    (perspective-коррекция из демо тиммейта в этот пайплайн пока не
    портирована — независимый шаг, можно добавить отдельно при
    необходимости, см. README)."""
    if not apply_clahe_:
        return image_bgr
    return apply_clahe(image_bgr, clip=clip, tile=tile)
