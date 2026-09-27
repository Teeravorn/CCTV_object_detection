"""Predict the test set and write a submission in the sample_submission.csv format.

Usage:
  python predict.py --kind yolo   --weights runs/yolo26s/weights/best.pt
  python predict.py --kind rfdetr --weights runs/rfdetr_medium/checkpoint_best_total.pth
Output: results/submission_<kind>.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from common import RESULTS_DIR, SAMPLE_SUB, TEST_IMG_DIR, submission_id_to_file
from detectors import load_detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["yolo", "rfdetr"])
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO only")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sub_ids = pd.read_csv(SAMPLE_SUB)["image_id"].drop_duplicates().tolist()
    file_of = {sid: submission_id_to_file(sid) for sid in sub_ids}
    missing = [f for f in file_of.values() if not (TEST_IMG_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} test images not found, e.g. {missing[:3]}")

    det = load_detector(args.kind, args.weights, args.imgsz)
    preds = det.predict([TEST_IMG_DIR / f for f in file_of.values()])

    rows, empty = [], 0
    for sid in sub_ids:
        boxes = preds.get(file_of[sid], [])
        if not boxes:
            # keep every image in the file, like sample_submission's placeholder row
            boxes = [(0, 0.01, 0, 0, 1, 1)]
            empty += 1
        for cls, score, x1, y1, x2, y2 in boxes:
            rows.append((sid, cls, round(score, 5), round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)))

    out = pd.DataFrame(rows, columns=["image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"])
    out.insert(0, "id", range(len(out)))
    path = args.out or RESULTS_DIR / f"submission_{args.kind}.csv"
    path.parent.mkdir(exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8")
    print(f"{len(sub_ids)} images, {len(out)} rows ({empty} images without detections) -> {path}")


if __name__ == "__main__":
    main()
