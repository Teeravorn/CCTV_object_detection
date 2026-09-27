"""Convert train.csv into a YOLO dataset (for YOLO26) and a COCO dataset (for RF-DETR).

Both datasets use the exact same camera-based train/val split (see common.VAL_CAMS),
so the two models are compared on identical data.

Usage:  python prepare_data.py
"""
import shutil
from collections import Counter

import pandas as pd
from PIL import Image

from common import (CLASS_NAMES, COCO_DIR, TRAIN_CSV, TRAIN_IMG_DIR, VAL_CAMS, YOLO_DIR, camera_of,
                    save_json)


def load_annotations():
    df = pd.read_csv(TRAIN_CSV)
    images = sorted(p.name for p in TRAIN_IMG_DIR.glob("*.jpg"))
    sizes = {}
    for name in images:
        with Image.open(TRAIN_IMG_DIR / name) as im:
            sizes[name] = im.size  # (w, h)

    missing = set(df["image_id"]) - set(images)
    if missing:
        print(f"[warn] {len(missing)} annotated images not found in train/, dropping them")
        df = df[~df["image_id"].isin(missing)]

    # Clip boxes to the image and drop degenerate ones
    w = df["image_id"].map(lambda n: sizes[n][0])
    h = df["image_id"].map(lambda n: sizes[n][1])
    df = df.assign(x1=df.x1.clip(0, w), x2=df.x2.clip(0, w), y1=df.y1.clip(0, h), y2=df.y2.clip(0, h))
    bad = (df.x2 - df.x1 < 1) | (df.y2 - df.y1 < 1)
    if bad.any():
        print(f"[warn] dropping {int(bad.sum())} degenerate boxes")
        df = df[~bad]
    return df, images, sizes


def write_yolo(df, split_images, sizes):
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)
    by_img = {k: g for k, g in df.groupby("image_id")}
    for split, names in split_images.items():
        img_dir = YOLO_DIR / "images" / split
        lbl_dir = YOLO_DIR / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for name in names:
            shutil.copy2(TRAIN_IMG_DIR / name, img_dir / name)
            W, H = sizes[name]
            lines = []
            if name in by_img:
                for r in by_img[name].itertuples():
                    cx, cy = (r.x1 + r.x2) / 2 / W, (r.y1 + r.y2) / 2 / H
                    bw, bh = (r.x2 - r.x1) / W, (r.y2 - r.y1) / H
                    lines.append(f"{r.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            # empty label file = background image (kept as negatives)
            (lbl_dir / (name[:-4] + ".txt")).write_text("\n".join(lines), encoding="utf-8")

    yaml = [f"path: {YOLO_DIR.as_posix()}", "train: images/train", "val: images/val", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)]
    (YOLO_DIR / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")


def coco_dict(df, names, sizes):
    images, anns = [], []
    by_img = {k: g for k, g in df.groupby("image_id")}
    for img_id, name in enumerate(names):
        W, H = sizes[name]
        images.append({"id": img_id, "file_name": name, "width": W, "height": H})
        if name in by_img:
            for r in by_img[name].itertuples():
                bw, bh = float(r.x2 - r.x1), float(r.y2 - r.y1)
                anns.append({
                    "id": len(anns), "image_id": img_id, "category_id": int(r.class_id),
                    "bbox": [float(r.x1), float(r.y1), bw, bh], "area": bw * bh, "iscrowd": 0,
                })
    cats = [{"id": i, "name": n, "supercategory": "none"} for i, n in enumerate(CLASS_NAMES)]
    return {"images": images, "annotations": anns, "categories": cats}


def write_coco(df, split_images, sizes):
    if COCO_DIR.exists():
        shutil.rmtree(COCO_DIR)
    # RF-DETR expects train/ valid/ test/ ; test is a copy of valid (only used if run_test=True)
    for split, names in [("train", split_images["train"]), ("valid", split_images["val"]),
                         ("test", split_images["val"])]:
        out = COCO_DIR / split
        out.mkdir(parents=True)
        for name in names:
            shutil.copy2(TRAIN_IMG_DIR / name, out / name)
        save_json(coco_dict(df, names, sizes), out / "_annotations.coco.json")


def main():
    df, images, sizes = load_annotations()
    split_images = {
        "train": [n for n in images if camera_of(n) not in VAL_CAMS],
        "val": [n for n in images if camera_of(n) in VAL_CAMS],
    }
    for split, names in split_images.items():
        sub = df[df.image_id.isin(names)]
        cls = Counter(sub.class_id)
        print(f"{split:5s}: {len(names):4d} images, {len(sub):5d} boxes, "
              f"cams={sorted({camera_of(n) for n in names}, key=int)}")
        print("       per class:", {k: cls.get(k, 0) for k in range(len(CLASS_NAMES))})

    write_yolo(df, split_images, sizes)
    write_coco(df, split_images, sizes)
    print(f"YOLO dataset -> {YOLO_DIR}\nCOCO dataset -> {COCO_DIR}")


if __name__ == "__main__":
    main()
