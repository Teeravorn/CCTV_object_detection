"""Predict the test set and write a submission in the sample_submission.csv format.

Usage:
  python predict.py --weights runs/rfdetr_medium/checkpoint_best_total.pth
  python predict.py --weights runs/rfdetr_medium/checkpoint_best_total.pth --dedupe-iou 0.9
Output: results/submission_rfdetr.csv (or results/submission_rfdetr_dedupe0.9_s0.3.csv)
"""
import argparse
from pathlib import Path

import pandas as pd

from common import RESULTS_DIR, SAMPLE_SUB, TEST_IMG_DIR, submission_id_to_file
from detectors import RFDETRDetector, dedupe_cross_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--id-style", choices=["file", "sample"], default="file",
                    help="file: test file name as on Kaggle (1068_20260825_060110.jpg) - scored 0 with 'sample'; "
                         "sample: the long dataset_..._annotated_... name from sample_submission.csv")
    ap.add_argument("--dedupe-iou", type=float, default=None,
                    help="drop a box when a higher-scoring box of another class overlaps it above this IoU "
                         "(e.g. 0.9); probes whether the test labels contain Car/Truck-style double labels")
    ap.add_argument("--dedupe-min-score", type=float, default=0.3,
                    help="only boxes scoring at least this are dropped; lower-score alternative classes, which "
                         "RF-DETR always emits, stay so the A/B comparison isolates confident double predictions")
    args = ap.parse_args()

    sub_ids = pd.read_csv(SAMPLE_SUB)["image_id"].drop_duplicates().tolist()
    file_of = {sid: submission_id_to_file(sid) for sid in sub_ids}
    missing = [f for f in file_of.values() if not (TEST_IMG_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} test images not found, e.g. {missing[:3]}")

    det = RFDETRDetector(args.weights)
    preds = det.predict([TEST_IMG_DIR / f for f in file_of.values()])
    if args.dedupe_iou is not None:
        before = sum(map(len, preds.values()))
        preds = {f: dedupe_cross_class(b, args.dedupe_iou, args.dedupe_min_score) for f, b in preds.items()}
        print(f"dedupe IoU>{args.dedupe_iou}, score>={args.dedupe_min_score}: removed {before - sum(map(len, preds.values()))} of {before} boxes")

    rows, empty = [], 0
    for sid in sub_ids:
        boxes = preds.get(file_of[sid], [])
        if not boxes:
            # competition rules: no rows (and no placeholder/dummy boxes) for images without detections
            empty += 1
        image_id = file_of[sid] if args.id_style == "file" else sid
        for cls, score, x1, y1, x2, y2 in boxes:
            rows.append((image_id, cls, round(score, 5), round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)))

    out = pd.DataFrame(rows, columns=["image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"])
    out.insert(0, "id", range(len(out)))
    suffix = f"_dedupe{args.dedupe_iou:g}_s{args.dedupe_min_score:g}" if args.dedupe_iou is not None else ""
    path = args.out or RESULTS_DIR / f"submission_rfdetr{suffix}.csv"
    path.parent.mkdir(exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8")
    print(f"{len(sub_ids)} images, {len(out)} rows ({empty} images without detections) -> {path}")


if __name__ == "__main__":
    main()
