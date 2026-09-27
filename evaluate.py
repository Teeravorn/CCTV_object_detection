"""Compare trained YOLO26 and RF-DETR on the held-out validation cameras.

Both models are scored by the same pycocotools evaluator on the same images, with conf>=0.001.

Usage:
  python evaluate.py --yolo runs/yolo26s/weights/best.pt --rfdetr runs/rfdetr_medium/checkpoint_best_total.pth
Outputs results/comparison.json and results/comparison.md
"""
import argparse
import math
from pathlib import Path

from common import CLASS_NAMES, COCO_DIR, RESULTS_DIR, coco_evaluate, save_json
from detectors import load_detector, timed_predict


def fmt(x):
    return "  -  " if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", type=Path, default=None)
    ap.add_argument("--yolo-imgsz", type=int, default=640)
    ap.add_argument("--rfdetr", type=Path, default=None)
    args = ap.parse_args()

    val_paths = sorted((COCO_DIR / "valid").glob("*.jpg"))
    print(f"Evaluating on {len(val_paths)} validation images")

    results = {}
    for kind, weights in [("yolo", args.yolo), ("rfdetr", args.rfdetr)]:
        if weights is None:
            continue
        det = load_detector(kind, weights, args.yolo_imgsz)
        preds, ms = timed_predict(det, val_paths)
        m = coco_evaluate(preds)
        m["ms_per_image"] = ms
        m["weights"] = str(weights)
        results[det.name] = m
        print(f"{det.name:8s} mAP50-95={m['mAP50-95']:.4f}  mAP50={m['mAP50']:.4f}  {ms:.1f} ms/img")
        del det

    if not results:
        ap.error("pass --yolo and/or --rfdetr")

    names = list(results)
    lines = ["| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for key in ["mAP50-95", "mAP50", "mAP75", "AP_small"]:
        lines.append(f"| {key} | " + " | ".join(fmt(results[n][key]) for n in names) + " |")
    lines.append("| ms / image | " + " | ".join(f"{results[n]['ms_per_image']:.1f}" for n in names) + " |")
    lines += ["", "Per-class mAP50-95", "", "| class | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for c in CLASS_NAMES:
        lines.append(f"| {c} | " + " | ".join(fmt(results[n]["per_class"].get(c, {}).get("mAP50-95"))
                                               for n in names) + " |")
    if len(names) == 2:
        a, b = names
        winner = a if results[a]["mAP50-95"] >= results[b]["mAP50-95"] else b
        lines += ["", f"**Better on mAP50-95: {winner}**"]

    table = "\n".join(lines)
    print("\n" + table)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "comparison.md").write_text(table + "\n", encoding="utf-8")
    save_json(results, RESULTS_DIR / "comparison.json")
    print(f"\nSaved -> {RESULTS_DIR / 'comparison.md'}")


if __name__ == "__main__":
    main()
