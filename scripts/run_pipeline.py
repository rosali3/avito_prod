#!/usr/bin/env python
"""
CLI: прогнать пайплайн (предобработка + RF-DETR + мебель-заглушка + OCR)
на одном изображении, сохранить результат.

Запуск:
    python scripts/run_pipeline.py --image path/to/plan.jpg --out-dir result/
    python scripts/run_pipeline.py --image plan.jpg --no-clahe --no-ocr
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.pipeline import run_pipeline  # noqa: E402
from pipeline.rfdetr_infer import CLASS_NAMES  # noqa: E402

PALETTE = {
    1: (255, 99, 71), 2: (60, 179, 113), 3: (65, 105, 225), 4: (255, 215, 0),
    5: (238, 130, 238), 6: (128, 128, 128), 7: (255, 140, 0),
}


def overlay_semantic_mask(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    out = image_bgr.astype(np.float32).copy()
    for cid, color in PALETTE.items():
        m = mask == cid
        if not m.any():
            continue
        bgr = np.array(color[::-1], dtype=np.float32)
        out[m] = out[m] * (1 - alpha) + bgr * alpha
    return out.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="путь к фото плана")
    ap.add_argument("--out-dir", default="pipeline_out", help="куда сохранить результат")
    ap.add_argument("--model", choices=["rfdetr", "unet"], default="rfdetr",
                     help="rfdetr (рекомендуется, лучше на UGC) или unet (+Canny room-fill)")
    ap.add_argument("--no-clahe", action="store_true", help="отключить CLAHE-предобработку")
    ap.add_argument("--no-ocr", action="store_true", help="отключить OCR площадей")
    ap.add_argument("--no-unet-room-fill", action="store_true",
                     help="для --model unet: отключить Canny room-fill постобработку")
    ap.add_argument("--rfdetr-threshold", type=float, default=0.15)
    args = ap.parse_args()

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise SystemExit(f"не читается изображение: {args.image}")

    result = run_pipeline(
        image_bgr,
        model=args.model,
        unet_room_fill=not args.no_unet_room_fill,
        apply_clahe=not args.no_clahe,
        rfdetr_threshold=args.rfdetr_threshold,
        run_ocr=not args.no_ocr,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overlay = overlay_semantic_mask(image_bgr, result["semantic_mask"])
    cv2.imwrite(str(out_dir / "overlay.png"), overlay)

    summary = {
        "n_instances": len(result["instances"]),
        "instances": [
            {"class_name": i["class_name"], "score": round(i["score"], 3), "bbox": i["bbox"]}
            for i in result["instances"]
        ],
        "furniture": result["furniture"],  # заглушка — всегда []
        "area_labels": [
            {"text": a["text"], "value_m2": a["value_m2"], "conf": round(a["conf"], 3)}
            for a in result["area_labels"]
        ],
    }
    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.model == "rfdetr":
        print(f"Найдено инстансов: {len(result['instances'])}")
        for cid, name in CLASS_NAMES.items():
            n = sum(1 for i in result["instances"] if i["category_id"] == cid)
            if n:
                print(f"  {name}: {n}")
    else:
        print("Классы на semantic-маске (UNet):")
        for cid, name in CLASS_NAMES.items():
            px = int((result["semantic_mask"] == cid).sum())
            if px:
                print(f"  {name}: {px} px")
    print(f"Подписей площади (OCR): {len(result['area_labels'])}")
    print(f"-> {out_dir / 'overlay.png'}")
    print(f"-> {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()
