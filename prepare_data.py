"""Convert train.csv into a COCO dataset for RF-DETR.

Train/val are split by camera (see common.VAL_CAMS) so validation mimics the unseen test cameras.

Usage:  python prepare_data.py                  # baseline
        python prepare_data.py --oversample     # repeat train images that contain rare classes
"""
import argparse
import math
import shutil
from collections import Counter

import pandas as pd
from PIL import Image

from common import CLASS_NAMES, COCO_DIR, PREPARE_INFO, TRAIN_CSV, TRAIN_IMG_DIR, VAL_CAMS, camera_of, save_json

# duplicated image name -> original file in TRAIN_IMG_DIR (filled by oversample())
SOURCE_OF = {}


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


def oversample(df, names, sizes, thresh, max_repeat):
    """Repeat-factor sampling (LVIS): class c with image frequency f_c gets r_c = max(1, sqrt(thresh / f_c));
    each image is repeated round(max r_c over its classes) times, capped at max_repeat.

    mAP50 weighs every class equally, so rare classes (Tuktuk, Van, Pickup, Songthaew) matter as much as Car.
    Copies are exact (same image, same boxes); on-the-fly augmentation makes them differ during training.
    """
    classes_of = df[df.image_id.isin(names)].groupby("image_id").class_id.apply(set).to_dict()
    img_freq = Counter(c for cs in classes_of.values() for c in cs)
    r_class = {c: max(1.0, math.sqrt(thresh / (img_freq[c] / len(names)))) for c in img_freq}

    new_names, new_rows = [], []
    for name in names:
        reps = round(min(max_repeat, max((r_class[c] for c in classes_of.get(name, ())), default=1.0)))
        for k in range(1, reps):
            dup = f"{name[:-4]}__rep{k}.jpg"
            SOURCE_OF[dup] = name
            sizes[dup] = sizes[name]
            new_names.append(dup)
            new_rows.append(df[df.image_id == name].assign(image_id=dup))
    print(f"oversample (thresh={thresh}, max_repeat={max_repeat}): +{len(new_names)} train images; repeat per class:",
          {CLASS_NAMES[c]: round(r_class[c], 2) for c in sorted(r_class)})
    return names + new_names, pd.concat([df, *new_rows], ignore_index=True)


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
    # RF-DETR only reads test/ when run_test=True, which train_rfdetr.py keeps off
    for split, names in [("train", split_images["train"]), ("valid", split_images["val"])]:
        out = COCO_DIR / split
        out.mkdir(parents=True)
        for name in names:
            shutil.copy2(TRAIN_IMG_DIR / SOURCE_OF.get(name, name), out / name)
        save_json(coco_dict(df, names, sizes), out / "_annotations.coco.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oversample", action="store_true", help="repeat train images containing rare classes")
    ap.add_argument("--os-thresh", type=float, default=0.3,
                    help="classes present in fewer than this fraction of train images get repeated")
    ap.add_argument("--os-max-repeat", type=int, default=4)
    args = ap.parse_args()

    df, images, sizes = load_annotations()
    split_images = {
        "train": [n for n in images if camera_of(n) not in VAL_CAMS],
        "val": [n for n in images if camera_of(n) in VAL_CAMS],
    }
    if args.oversample:  # train only: validation must stay a faithful estimate
        split_images["train"], df = oversample(df, split_images["train"], sizes, args.os_thresh, args.os_max_repeat)
    for split, names in split_images.items():
        sub = df[df.image_id.isin(names)]
        cls = Counter(sub.class_id)
        print(f"{split:5s}: {len(names):4d} images, {len(sub):5d} boxes, "
              f"cams={sorted({camera_of(n) for n in names}, key=int)}")
        print("       per class:", {k: cls.get(k, 0) for k in range(len(CLASS_NAMES))})

    write_coco(df, split_images, sizes)
    save_json({"oversample": args.oversample,
               "os_thresh": args.os_thresh if args.oversample else None,
               "os_max_repeat": args.os_max_repeat if args.oversample else None}, PREPARE_INFO)
    print(f"COCO dataset -> {COCO_DIR}")


if __name__ == "__main__":
    main()
