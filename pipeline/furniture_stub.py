"""
ЗАГЛУШКА мебельного детектора — реальная модель НЕ готова к использованию.

Статус (см. claude_instseg_compare/docs/furniture_experiments_log.md):
  - На своём домене (SFPI/FloorPlanCAD val) модель работает отлично
    (mAP50(mask)=0.992, confidence 0.87-0.99).
  - На UGC (реальные фото — целевой домен ЭТОГО пайплайна) модель
    визуально СЛОМАНА: коллапсирует в один мажоритарный класс (чаще
    всего "sink") почти на всех детекциях, включая подписи площади,
    номера комнат, водяной знак. Классический domain-gap collapse,
    аналогичный тому, что наблюдался у недообученного RF-DETR на его
    собственной валидации.
  - Реранкер (логика "мебель корректирует room-тип") тоже не написан.

Поэтому реальный вызов инференса здесь ЗАКОММЕНТИРОВАН — подключать
только после того, как модель будет дообучена/переделана специально
под реальные фото и заново провалидирована на UGC с количественными
метриками (сейчас их нет вообще, только визуальная проверка).
"""
from __future__ import annotations

import numpy as np


def run_furniture_detector(image_bgr: np.ndarray) -> list[dict]:
    """Заглушка: всегда возвращает пустой список инстансов.

    Реальный вызов (когда модель будет готова к использованию на UGC):

        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download
        _ckpt = hf_hub_download(repo_id="<TODO: furniture checkpoint repo>",
                                 filename="furniture_yolo11n_cont2/best.pt")
        _model = YOLO(_ckpt)
        results = _model.predict(image_bgr, conf=0.25)
        return [
            {"category_id": int(box.cls), "class_name": _model.names[int(box.cls)],
             "score": float(box.conf), "bbox": box.xywh.tolist()}
            for box in results[0].boxes
        ]
    """
    return []
