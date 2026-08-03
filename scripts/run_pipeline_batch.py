#!/usr/bin/env python
"""
Прогнать pipeline на всех картинках из папки (модель грузится один раз,
не на каждое изображение — быстрее, чем звать run_pipeline.py в цикле).

⚠️ ИЗВЕСТНАЯ ПРОБЛЕМА (не починено): при --ocr (по умолчанию включён)
повторное использование ОДНОГО и того же PaddleOCR-движка на десятках
картинок подряд в одном процессе иногда роняет процесс в Segmentation
fault (похоже на утечку/порчу состояния в paddle-бэкенде, воспроизведено
на Windows) — без трейсбека, Python try/except его не ловит. Обходной
путь, которым реально пользовались для батч-прогона всех 33 UGC test
картинок — вызывать run_pipeline.py ОТДЕЛЬНЫМ процессом на каждую
картинку (в bash-цикле), это переживает падение отдельного процесса.
Для --no-ocr (только RF-DETR/UNet) этот скрипт должен быть безопасен —
краш наблюдался именно в PaddleOCR-пути.

Запуск:
    python scripts/run_pipeline_batch.py --images-dir path/to/images --out-dir results/rfdetr
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.pipeline import run_pipeline  # noqa: E402
from run_pipeline import overlay_semantic_mask, append_area_footer, CLASS_NAMES  # noqa: E402

EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--model", choices=["rfdetr", "unet"], default="rfdetr")
    ap.add_argument("--no-clahe", action="store_true")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--rfdetr-threshold", type=float, default=0.15)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path("results") / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    images_dir = Path(args.images_dir)
    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in EXTS)
    print(f"[batch] {len(files)} картинок в {images_dir}")

    for i, path in enumerate(files):
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            print(f"  [skip] не читается: {path.name}")
            continue

        result = run_pipeline(
            image_bgr,
            model=args.model,
            apply_clahe=not args.no_clahe,
            rfdetr_threshold=args.rfdetr_threshold,
            run_ocr=not args.no_ocr,
        )

        overlay = overlay_semantic_mask(image_bgr, result["semantic_mask"])
        overlay = append_area_footer(overlay, result["area_labels"])
        stem = path.stem
        cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay)

        summary = {
            "n_instances": len(result["instances"]),
            "instances": [
                {"class_name": inst["class_name"], "score": round(inst["score"], 3), "bbox": inst["bbox"]}
                for inst in result["instances"]
            ],
            "furniture": result["furniture"],
            "area_labels": [
                {"text": a["text"], "value_m2": a["value_m2"], "conf": round(a["conf"], 3)}
                for a in result["area_labels"]
            ],
        }
        with open(out_dir / f"{stem}_result.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"  [{i + 1}/{len(files)}] {stem}: {len(result['instances'])} инстансов, "
              f"{len(result['area_labels'])} площадей -> {out_dir / f'{stem}_overlay.png'}")

    print(f"[batch] готово -> {out_dir}")


if __name__ == "__main__":
    main()
