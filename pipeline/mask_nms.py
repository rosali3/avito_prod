"""
NMS по маскам (не по боксам): если две предсказанные маски на одной картинке
пересекаются сильнее iou_thresh, оставляем только ту, у которой выше score.
Кросс-классовый (не только внутри одного класса) — на низком score-threshold
RF-DETR часто выдаёт дублирующие маски разных классов почти в одном месте.

Портировано без изменений из claude_instseg_compare/eval/mask_nms.py
(основной research-репозиторий, там же полная методология в
docs/room_postprocessing_experiments.md).
"""
from __future__ import annotations

from pycocotools import mask as mask_utils


def _to_rle(seg, h: int, w: int) -> dict:
    if isinstance(seg, dict):
        return seg
    rles = mask_utils.frPyObjects(seg, h, w)
    return mask_utils.merge(rles)


def mask_nms(predictions: list[dict], img_wh: dict[int, tuple[int, int]], iou_thresh: float = 0.5) -> list[dict]:
    """predictions: список dict(image_id, category_id, score, segmentation).
    img_wh: {image_id: (height, width)}. Возвращает отфильтрованный список."""
    by_img: dict[int, list[int]] = {}
    for i, p in enumerate(predictions):
        by_img.setdefault(p["image_id"], []).append(i)

    keep_idx: set[int] = set()
    for img_id, idxs in by_img.items():
        h, w = img_wh[img_id]
        idxs_sorted = sorted(idxs, key=lambda i: predictions[i].get("score", 1.0), reverse=True)
        rles = [_to_rle(predictions[i]["segmentation"], h, w) for i in idxs_sorted]

        kept_local: list[int] = []
        for local_i in range(len(idxs_sorted)):
            suppressed = False
            for local_j in kept_local:
                iou = mask_utils.iou([rles[local_i]], [rles[local_j]], [0])[0][0]
                if iou > iou_thresh:
                    suppressed = True
                    break
            if not suppressed:
                kept_local.append(local_i)
                keep_idx.add(idxs_sorted[local_i])

    return [p for i, p in enumerate(predictions) if i in keep_idx]
