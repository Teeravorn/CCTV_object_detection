"""Compare trained RF-DETR checkpoints on the held-out validation cameras.

Every checkpoint is scored by the same pycocotools evaluator on the same images, with conf>=0.001.
The headline metric is mAP50, matching the Kaggle leaderboard (pycocotools, IoU 0.5, maxDets 100).

Usage:
  python evaluate.py --rfdetr runs/rfdetr_medium/checkpoint_best_total.pth [runs/rfdetr_medium_aug/... ...]
Outputs results/comparison.json and results/comparison.md
"""
import argparse
import math
from pathlib import Path

from common import CLASS_NAMES, COCO_DIR, RESULTS_DIR, coco_evaluate, save_json
from detectors import RFDETRDetector, timed_predict


def fmt(x):
    return "  -  " if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rfdetr", type=Path, nargs="+", required=True, help="one or more checkpoints")
    args = ap.parse_args()

    val_paths = sorted((COCO_DIR / "valid").glob("*.jpg"))
    print(f"Evaluating on {len(val_paths)} validation images")

    results = {}
    for weights in args.rfdetr:
        name = weights.parent.name  # run dir, e.g. rfdetr_medium_aug
        det = RFDETRDetector(weights)
        preds, ms = timed_predict(det, val_paths)
        m = coco_evaluate(preds)
        m["ms_per_image"] = ms
        m["weights"] = str(weights)
        results[name] = m
        print(f"{name:20s} mAP50={m['mAP50']:.4f} (Kaggle metric)  mAP50-95={m['mAP50-95']:.4f}  {ms:.1f} ms/img")
        del det

    names = list(results)
    lines = ["| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for key in ["mAP50", "mAP50-95", "mAP75", "AP_small"]:
        lines.append(f"| {key} | " + " | ".join(fmt(results[n][key]) for n in names) + " |")
    lines.append("| ms / image | " + " | ".join(f"{results[n]['ms_per_image']:.1f}" for n in names) + " |")
    lines += ["", "Per-class AP50", "", "| class | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for c in CLASS_NAMES:
        lines.append(f"| {c} | " + " | ".join(fmt(results[n]["per_class"].get(c, {}).get("mAP50"))
                                               for n in names) + " |")
    if len(names) > 1:
        winner = max(names, key=lambda n: results[n]["mAP50"])
        lines += ["", f"**Best on mAP50 (Kaggle metric): {winner}**"]

    table = "\n".join(lines)
    print("\n" + table)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "comparison.md").write_text(table + "\n", encoding="utf-8")
    save_json(results, RESULTS_DIR / "comparison.json")
    print(f"\nSaved -> {RESULTS_DIR / 'comparison.md'}")


if __name__ == "__main__":
    main()
