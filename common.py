"""Shared paths, split definition, and COCO-style evaluation for YOLO26 vs RF-DETR."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN_IMG_DIR = ROOT / "data_source" / "train"
TEST_IMG_DIR = ROOT / "data_source" / "test"
TRAIN_CSV = ROOT / "train.csv"
SAMPLE_SUB = ROOT / "sample_submission.csv"

DATA_DIR = ROOT / "data"
YOLO_DIR = DATA_DIR / "yolo"
COCO_DIR = DATA_DIR / "coco"
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"

NUM_CLASSES = 8
# From the Kaggle Data page (class_id -> name)
CLASS_NAMES = ["Car", "Motorcycle", "Bus", "Truck", "Tuktuk", "Van", "Pickup", "Songthaew"]

# Test cameras (1068, 1072, 1192, 1439, 227) never appear in train, so validation must also be
# unseen cameras. These three cover all 8 classes (incl. rare class 4/5/6/7) and are ~20% of images.
VAL_CAMS = {"1427", "232", "244"}

# Ground truth used by evaluate.py (written by prepare_data.py)
VAL_GT_JSON = COCO_DIR / "valid" / "_annotations.coco.json"


def camera_of(filename: str) -> str:
    return filename.split("_", 1)[0]


_SUB_ID_RE = re.compile(r"^dataset_(\d+)_.*_(\d{8}_\d{6})\.jpg$")


def submission_id_to_file(image_id: str) -> str:
    """'dataset_1068_annotated_..._20260825_060110.jpg' -> '1068_20260825_060110.jpg'"""
    m = _SUB_ID_RE.match(image_id)
    if not m:
        raise ValueError(f"Unrecognised submission image_id: {image_id}")
    return f"{m.group(1)}_{m.group(2)}.jpg"


def coco_evaluate(predictions: dict, gt_json: Path = VAL_GT_JSON) -> dict:
    """Evaluate predictions with pycocotools.

    predictions: {file_name: [(class_id, score, x1, y1, x2, y2), ...]}
    Kaggle scores mAP@.5 ("mAP50"). Returns overall mAP@[.5:.95], mAP@.5, mAP@.75, AP_small and
    per-class mAP@[.5:.95] / mAP@.5.
    """
    import contextlib
    import io

    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_json))
    name_to_id = {img["file_name"]: img["id"] for img in gt.dataset["images"]}

    dets = []
    for fname, boxes in predictions.items():
        img_id = name_to_id[fname]
        for cls, score, x1, y1, x2, y2 in boxes:
            dets.append({
                "image_id": img_id,
                "category_id": int(cls),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
            })
    if not dets:
        return {"mAP50-95": 0.0, "mAP50": 0.0, "mAP75": 0.0, "AP_small": 0.0, "per_class": {}}

    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets)
        ev = COCOeval(gt, dt, iouType="bbox")
        # Kaggle: pycocotools, default COCO maxDets=100 (applied per image per class)
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    # precision: [T(iou), R, K(class), A(area), M(maxDets)]
    prec = ev.eval["precision"][:, :, :, 0, -1]
    per_class = {}
    for k, cat_id in enumerate(ev.params.catIds):
        p_all = prec[:, :, k]
        p50 = prec[0, :, k]
        per_class[CLASS_NAMES[cat_id]] = {
            "mAP50-95": float(np.mean(p_all[p_all > -1])) if (p_all > -1).any() else float("nan"),
            "mAP50": float(np.mean(p50[p50 > -1])) if (p50 > -1).any() else float("nan"),
        }
    return {
        "mAP50-95": float(ev.stats[0]),
        "mAP50": float(ev.stats[1]),
        "mAP75": float(ev.stats[2]),
        "AP_small": float(ev.stats[3]),
        "per_class": per_class,
    }


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
