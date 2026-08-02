# Обучение моделей и воспроизводимость

Этот репозиторий — **inference-only**, специально без train-кода: чтобы
Docker-образ для деплоя оставался лёгким (не тащить torch+CUDA+
mmdetection+ultralytics+transformers одновременно ради обучения,
которое здесь не происходит).

## Где реально лежит train-пайплайн

Полный код обучения всех 9 сравниваемых моделей (RF-DETR-Seg, RF-DETR
fullaug, YOLO11m-seg (+fullaug), Mask R-CNN, SegFormer, UNet, SAM
zero-shot/fine-tuned), подготовка данных (`data_prep/*`, детерминированный
80/20 split seed=42), вся методология оценки (ignore-regions,
room=living∪bedroom матчинг, mask-NMS, confidence-threshold sweep) и
полная история экспериментов с пост-процессингом — в research-репозитории:

**https://github.com/rosali3/avito_floorplan_segmentation-**

Ключевые документы там:
- `docs/experiments_log.md` — журнал обучения всех моделей: гиперпараметры,
  на каких эпохах остановлены и почему, все метрики (UGC test + собственный
  held-out val на ResPlan/CubiCasa), confidence-threshold sweep.
- `docs/room_postprocessing_experiments.md` — все эксперименты с
  пост-процессингом (Canny-fill, squareness-фильтр, mask-NMS) и полные
  таблицы метрик по всем 9 моделям.
- `docs/furniture_experiments_log.md` — мебельный детектор и OCR (обучение,
  датасеты, известные проблемы).

## Чекпоинты

Все финальные чекпоинты (включая RF-DETR-Seg, используемый этим
пайплайном) выложены на HuggingFace Hub:
**https://huggingface.co/nabiullina-dstu/avito-floorplan-checkpoints**

Этот пайплайн скачивает чекпоинт RF-DETR оттуда автоматически при первом
запуске (см. `pipeline/rfdetr_infer.py`) — воспроизводить обучение для
запуска инференса не нужно.
