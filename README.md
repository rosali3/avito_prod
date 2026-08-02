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

### Docker Hub (готовый образ, разворачивать без сборки)

🐳 https://hub.docker.com/r/nabiullinastudy/floorplan-inference

**1. Скачать образ** (один раз, ~10.7 ГБ — включает torch/rfdetr/paddleocr):
```bash
docker pull nabiullinastudy/floorplan-inference:latest
```

**2. Запустить на своей картинке:**
```bash
docker run --rm \
    -v /путь/к/твоей/картинке.jpg:/data/plan.jpg \
    -v /путь/куда/сохранить/result:/app/result \
    nabiullinastudy/floorplan-inference --image /data/plan.jpg --out-dir /app/result
```
Чекпоинт модели (RF-DETR или UNet) скачается автоматически с HuggingFace
Hub при первом запуске и закэшируется — чтобы не качать заново при
каждом `docker run`, смонтируй volume под кэш:
```bash
docker run --rm \
    -v /путь/к/картинке.jpg:/data/plan.jpg \
    -v /путь/к/result:/app/result \
    -v hf_cache:/root/.cache/huggingface \
    nabiullinastudy/floorplan-inference --image /data/plan.jpg --out-dir /app/result
```

**3. Все доступные флаги** (после `--image ... --out-dir ...`):

| Флаг | По умолчанию | Что делает |
|---|---|---|
| `--model {rfdetr,unet}` | `rfdetr` | какая модель сегментации (RF-DETR рекомендуется, лучше на UGC) |
| `--rfdetr-threshold N` | `0.15` | confidence-порог для RF-DETR (per-model оптимум по room F1) |
| `--no-clahe` | выкл (CLAHE применяется) | отключить CLAHE-предобработку |
| `--no-ocr` | выкл (OCR применяется) | отключить OCR подписей площади (ускоряет прогон) |
| `--no-unet-room-fill` | выкл (fill применяется) | только с `--model unet`: отключить Canny room-fill постобработку |

Пример с UNet вместо RF-DETR:
```bash
docker run --rm -v /путь/к/картинке.jpg:/data/plan.jpg -v /путь/к/result:/app/result \
    nabiullinastudy/floorplan-inference --image /data/plan.jpg --out-dir /app/result --model unet
```

## Как проверить руками, что образ работает

После запуска (любым из способов выше) в `result/` появятся:
- `overlay.png` — фото с наложенной цветной маской предсказаний
- `result.json` — список инстансов (класс, confidence, bbox), опционально
  подписи площади (OCR)

Что смотреть, чтобы понять "работает или нет":
- В консоли строка `Найдено инстансов: N` — для реального фото плана
  должно быть **больше нуля**, обычно десятки (проверочный прогон дал 41
  инстанс с разумным распределением по классам). Если 0 или max
  confidence <0.1 — что-то сломано (см. "Известная проблема" ниже —
  такое уже было и починено).
- Открой `overlay.png` глазами — маска должна визуально лежать на
  комнатах/стенах, а не быть пустой или мусорной.

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

## Известная проблема (ИСПРАВЛЕНО)

Изначально RF-DETR давал нулевые/мусорные предсказания (max confidence
~0.07-0.09) — потратили немало времени на диагностику (проверили и
отклонили гипотезы про `numpy 2.x`/`paddlepaddle`-конфликт, порчу файла
при переносе, версии `rfdetr`/`torch`, CPU vs GPU). **Реальная причина
оказалась проще**: в `rfdetr_infer.py` был указан не тот файл весов —
`checkpoint_best_ema.pth` вместо `checkpoint_best_regular.pth`. EMA-веса
для этого конкретного чекпоинта не успели сойтись (RF-DETR обучен всего
4-5 эпох) и дают вырожденный результат, тогда как "regular" веса — рабочие
(162 инстанса на тестовом фото, max score 0.478, совпадает с
архивным результатом эксперимента). Исправлено, залито недостающий
`checkpoint_best_regular.pth` на HuggingFace Hub для базовой модели,
**подтверждено рабочим end-to-end в Docker на school1** (41 инстанс с
разумным распределением по классам на тестовом фото).

## Что НЕ включено

- **Perspective-коррекция** (keystone-распрямление из демо тиммейта) — не
  портирована, независимый шаг, можно добавить отдельно при необходимости.
- **Постобработка "§19"** (denoise/wall_close/opening_filt/... из демо
  тиммейта) — по их же данным почти не влияет на чистые маски RF-DETR
  (0.832→0.828, разрабатывалась под шумный U-Net) — не подключена
  намеренно, а не забыта.
- **Train-пайплайн** — см. `TRAINING.md`.
