# floorplan-inference

«А где у вас туалет?»: инстанс-сегментация комнат на поэтажных планах в
условиях domain gap между синтетикой (ResPlan/CubiCasa5K) и реальными
фото объявлений (Avito). Восемь конфигураций шести архитектур
(RF-DETR-Seg, YOLO-seg, Mask R-CNN, SegFormer, SAM, UNet) сравнены по
единой методологии на собственноручно размеченном UGC-датасете из 33
фото. Этот репозиторий — итоговый лёгкий inference-пайплайн: CLAHE
предобработка → сегментация комнат → (задел) детекция мебели → OCR
подписанных площадей.

📄 **Полный отчёт**: [`Avito1_Ridiger_Nabiullina_report.pdf`](Avito1_Ridiger_Nabiullina_report.pdf)

Это **inference-only** репозиторий — облегчённый, для деплоя/демо. Полное
исследование (сравнение 9 моделей, вся методология оценки, эксперименты
с пост-процессингом, threshold sweep) и **весь train-пайплайн** — в
отдельном research-репозитории, см. [`TRAINING.md`](TRAINING.md).

## Смежные репозитории

- **Код с экспериментами модели**: https://github.com/rosali3/avito_floorplan_segmentation-
- **Код с экспериментами по данным**: https://github.com/VelmorSdfg/ResPlan_Dataset

## Пайплайн

```
фото плана
  → preprocessing.py   CLAHE (clip=2.0, tile=8x8) — включено по умолчанию,
                        калибровано на UGC-подобных (тусклых) фото
  → --model rfdetr (по умолчанию, рекомендуется):
      rfdetr_infer.py   RF-DETR-Seg (Medium), чекпоинт с HuggingFace Hub,
                        threshold=0.15 (per-model оптимум по room F1),
                        mask-NMS iou=0.5
  → --model unet (альтернатива):
      unet_infer.py     UNet-simple, чекпоинт с HuggingFace Hub
      room_fill.py       + Canny room-fill постобработка (room F1 0.568->0.588
                        на UGC test — единственный сценарий, где этот
                        пост-процессинг реально помогает, см. README
                        research-репозитория)
  → furniture_stub.py   ЗАГЛУШКА — реальный вызов закомментирован,
                        см. файл и раздел "Статус мебельной модели" ниже
  → ocr.py              PaddleOCR (без предобработки — CLAHE ей вредит)
                        для подписей площади комнат, слабый опциональный сигнал
  → pipeline.py          собирает всё в один результат (JSON + overlay PNG)
```

RF-DETR рекомендуется по умолчанию (лучше на UGC test: segm AP@50=0.277
против UNet 0.050, см. research-репозиторий) — UNet добавлен как
альтернатива/для сравнения, не как основной путь.

## Запуск

### Локально
```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --image plan.jpg --out-dir result/
# или UNet + Canny room-fill вместо RF-DETR:
python scripts/run_pipeline.py --image plan.jpg --model unet --out-dir result/
```
Чекпоинт (RF-DETR либо UNet, в зависимости от `--model`) скачается
автоматически с HuggingFace Hub (`nabiullina-dstu/avito-floorplan-checkpoints`)
при первом запуске.

### Docker
```bash
docker build -t floorplan-inference .
docker run -v $(pwd)/plan.jpg:/data/plan.jpg -v $(pwd)/result:/app/result \
    -v hf_cache:/root/.cache/huggingface \
    floorplan-inference --image /data/plan.jpg --out-dir /app/result
```

## Статус мебельной модели

Мебельный детектор **не подключён** к пайплайну — вызов в `furniture_stub.py`
закомментирован, функция всегда возвращает `[]`. Причина: на своём домене
(SFPI/FloorPlanCAD) модель отличная (mAP50=0.992), но на реальных UGC-фото
визуально сломана (коллапс в один класс почти на каждой детекции). Полная
история и визуализации — в research-репозитории,
`docs/furniture_experiments_log.md`. Подключать реальный вызов только
после дообучения/переоценки на UGC с количественными метриками.

## OCR площадей

PaddleOCR без предобработки — по замерам на 67 ручных подписях площади
recall 38.8% (лучший результат из трёх сверенных движков, см.
`docs/furniture_experiments_log.md` в research-репозитории). Это слабый
опциональный сигнал, не источник истины — большинство подписей на плане
он не найдёт вообще.

## Известная проблема

RF-DETR даёт **нулевые/мусорные предсказания** (max confidence ~0.07-0.09
на картинках, где по сохранённому официальному результату исследования —
`claude_instseg_compare/output/rfdetr_seg/predictions/test_predictions.json`
— для ТОГО ЖЕ файла с ТЕМ ЖЕ чекпоинтом должно быть 162 инстанса, max
score 0.478). Собрано в Docker на school1 (Linux, чистое окружение,
`numpy<2`, тот же чекпоинт по хэшу совпадает с оригинальным) — баг
воспроизводится и там, значит **это не про numpy/paddlepaddle-конфликт**
(эта гипотеза проверена и отклонена). Файл изображения (`md5sum`) и
чекпоинта (размер) совпадают с исходными — не порча данных при переносе.
Текущая рабочая гипотеза: **версии `rfdetr`/`torch` не запинены** ни
здесь, ни в исходном research-репозитории (`pip install rfdetr torch`
без версий) — на момент оригинального инференса `pip` подтянул одни
версии, сейчас (в новом окружении) — другие, возможно с регрессией в
загрузке чекпоинта (в логе есть тревожный warning: "checkpoint lacks
args.num_queries / args.group_detr; falling back to flat slice ... may
scramble per-group query structure"). Не допроверено — детерминированность
подтверждена (5+ повторных запусков дают одинаковый неверный результат),
то есть это не случайность, а воспроизводимая проблема конкретно с
версией пакета. **Пайплайн НЕ протестирован end-to-end с рабочим
RF-DETR-инференсом.** UNet-путь (`--model unet`) тоже не проверен по той
же причине (не успели из-за приоритета на диагностику RF-DETR).

Перед первым реальным использованием: найти версии `rfdetr`/`torch`,
которыми изначально считался `test_predictions.json` (проверить
`pip freeze` в истории/логах research-репозитория, если сохранились),
запинить их явно в `requirements.txt`, и заново прогнать
`scripts/run_pipeline.py` на тестовом фото до полной проверки.

## Что НЕ включено

- **Perspective-коррекция** (keystone-распрямление из демо тиммейта) — не
  портирована, независимый шаг, можно добавить отдельно при необходимости.
- **Постобработка "§19"** (denoise/wall_close/opening_filt/... из демо
  тиммейта) — по их же данным почти не влияет на чистые маски RF-DETR
  (0.832→0.828, разрабатывалась под шумный U-Net) — не подключена
  намеренно, а не забыта.
- **Train-пайплайн** — см. `TRAINING.md`.
