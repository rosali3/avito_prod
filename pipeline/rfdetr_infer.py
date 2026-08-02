"""
RF-DETR-Seg (Medium) инференс — основная модель сегментации плана.

Чекпоинт: `rfdetr_seg/checkpoint_best_ema.pth` с HuggingFace Hub
(nabiullina-dstu/avito-floorplan-checkpoints) — тот же чекпоинт, что
и в основном research-репозитории (claude_instseg_compare), скачивается
лениво при первом запуске и кэшируется (~/.cache/huggingface).

Почему base, а не fullaug: на UGC test (реальные фото — целевой домен
этого пайплайна) base RF-DETR даёт segm AP@50=0.277 против 0.200 у
fullaug (см. claude_instseg_compare/docs/experiments_log.md) — fullaug
переобучился под искусственные аугментации коллеги, не совпадающие с
реальными UGC-дефектами.

Порог confidence=0.15 — не дефолтный 0.1 (тот был выбран для ЧЕСТНОГО
кросс-модельного сравнения в исследовании, не для продакшена): по
собственному threshold-sweep именно на 0.15 у RF-DETR максимальный
room F1 (0.746, room = living+bedroom) — см. experiments_log.md,
"Confidence-threshold sweep". Дальше — mask-NMS (iou=0.5, кросс-
классовый) — на низком пороге RF-DETR даёt много дублирующих масок
на одном объекте, без NMS metrics резко хуже (см.
room_postprocessing_experiments.md, "Mask-NMS для RF-DETR").

Таксономия (id 1-7, id 0 = background):
    1 living, 2 bedroom, 3 bathroom, 4 kitchen, 5 balcony, 6 wall, 7 opening
canonical_id = class_id_0based + 1 (модель обучена ровно на этих 7
классах в этом порядке, доп. маппинг не нужен).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .mask_nms import mask_nms

HF_REPO = "nabiullina-dstu/avito-floorplan-checkpoints"
# ВАЖНО: именно _regular, не _ema! EMA-веса для этого чекпоинта дают
# практически нулевой confidence (max ~0.08 на любой картинке) — EMA не
# успела сойтись за 4-5 эпох обучения (RF-DETR остановлен очень рано).
# Обнаружено сравнением с оригинальным test_predictions.json (там 162
# инстанса, max score 0.478 — воспроизводится только с _regular).
HF_FILENAME = "rfdetr_seg/checkpoint_best_regular.pth"
DEFAULT_THRESHOLD = 0.15
DEFAULT_NMS_IOU = 0.5

CLASS_NAMES = {1: "living", 2: "bedroom", 3: "bathroom", 4: "kitchen",
               5: "balcony", 6: "wall", 7: "opening"}

_model = None


def _get_checkpoint_path() -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)


def load_model():
    """Ленивая загрузка модели (один раз на процесс)."""
    global _model
    if _model is None:
        from rfdetr import RFDETRSegMedium
        ckpt_path = _get_checkpoint_path()
        _model = RFDETRSegMedium(pretrain_weights=ckpt_path)
    return _model


def run_rfdetr(image_bgr: np.ndarray, threshold: float = DEFAULT_THRESHOLD,
               nms_iou: float = DEFAULT_NMS_IOU) -> list[dict]:
    """Возвращает список инстансов после mask-NMS:
    [{"category_id": int, "class_name": str, "score": float,
      "mask": np.ndarray[H,W] bool, "bbox": [x,y,w,h]}, ...]
    """
    model = load_model()
    h, w = image_bgr.shape[:2]
    image_rgb = image_bgr[:, :, ::-1]
    pil_image = Image.fromarray(image_rgb)

    dets = model.predict(pil_image, threshold=threshold)
    n = len(dets.xyxy)
    has_mask = getattr(dets, "mask", None) is not None

    predictions = []
    for i in range(n):
        cid_0based = int(dets.class_id[i])
        canonical_id = cid_0based + 1
        if canonical_id not in CLASS_NAMES:
            continue  # редкий edge-case вне 7 классов, см. experiments_log.md
        x0, y0, x1, y1 = dets.xyxy[i].tolist()
        mask = np.asarray(dets.mask[i]) if has_mask else _bbox_to_mask(x0, y0, x1, y1, h, w)
        predictions.append({
            "image_id": 0,
            "category_id": canonical_id,
            "class_name": CLASS_NAMES[canonical_id],
            "score": float(dets.confidence[i]),
            "segmentation": _mask_to_rle(mask),
            "bbox": [x0, y0, x1 - x0, y1 - y0],
        })

    predictions = mask_nms(predictions, {0: (h, w)}, iou_thresh=nms_iou)

    from pycocotools import mask as mask_utils
    for p in predictions:
        p["mask"] = mask_utils.decode(p["segmentation"]).astype(bool)
        del p["segmentation"]
        del p["image_id"]
    return predictions


def _bbox_to_mask(x0, y0, x1, y1, h, w) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    xi0, yi0, xi1, yi1 = map(int, [x0, y0, x1, y1])
    mask[max(0, yi0):yi1, max(0, xi0):xi1] = 1
    return mask.astype(bool)


def _mask_to_rle(mask: np.ndarray) -> dict:
    from pycocotools import mask as mask_utils
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def assemble_semantic_mask(instances: list[dict], h: int, w: int) -> np.ndarray:
    """Инстансы -> плотная semantic-маска [H,W] (argmax по уверенности):
    рисуем по возрастанию confidence, так самая уверенная маска рисуется
    последней и "выигрывает" перекрытия. Тот же приём, что в
    floorplan_demo/gen_rfdetr_cache.py."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for inst in sorted(instances, key=lambda p: p["score"]):
        mask[inst["mask"]] = inst["category_id"]
    return mask
