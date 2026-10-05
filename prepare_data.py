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

    cleaning = {"csv_rows": len(df)}

    missing = df["image_id"].isin(set(df["image_id"]) - set(images))
    cleaning["dropped_rows_image_missing"] = int(missing.sum())
    if missing.any():
        print(f"[warn] {int(missing.sum())} boxes on images not found in train/, dropping them")
        df = df[~missing]

    # Clip boxes to the image and drop degenerate ones
    w = df["image_id"].map(lambda n: sizes[n][0])
    h = df["image_id"].map(lambda n: sizes[n][1])
    df = df.assign(x1=df.x1.clip(0, w), x2=df.x2.clip(0, w), y1=df.y1.clip(0, h), y2=df.y2.clip(0, h))
    bad = (df.x2 - df.x1 < 1) | (df.y2 - df.y1 < 1)
    cleaning["dropped_boxes_degenerate"] = int(bad.sum())
    if bad.any():
        print(f"[warn] dropping {int(bad.sum())} degenerate boxes")
        df = df[~bad]
    return df, images, sizes, cleaning


def drop_train_duplicates(df):
    """Drop rows repeated exactly (same image, class and box) in the train split only.

    A duplicate forces DETR's one-to-one matching to predict the same object twice. Validation keeps them so it
    still mirrors the Kaggle ground truth, which presumably has the same repeats. Near-identical boxes with a
    *different* class (mostly Car/Truck) are kept: they look systematic, so the test labels likely share them.
    """
    in_train = ~df["image_id"].map(camera_of).isin(VAL_CAMS)
    dup = in_train & df.duplicated(["image_id", "class_id", "x1", "y1", "x2", "y2"])
    print(f"dropping {int(dup.sum())} exact duplicate boxes from train")
    return df[~dup], int(dup.sum())


def split_stats(df, names):
    """Per-split counts for the report: images, empty images, cameras, boxes per class and per COCO size bucket."""
    sub = df[df.image_id.isin(names)]
    area = (sub.x2 - sub.x1) * (sub.y2 - sub.y1)
    cls = Counter(sub.class_id)
    return {
        "images": len(names),
        "images_without_boxes": len(set(names) - set(sub.image_id)),
        "cameras": sorted({camera_of(n) for n in names}, key=int),
        "boxes": len(sub),
        "boxes_per_class": {CLASS_NAMES[k]: cls.get(k, 0) for k in range(len(CLASS_NAMES))},
        # COCO buckets, as used by AP_small / AP_medium / AP_large
        "boxes_per_size": {"small (<32^2 px)": int((area < 32 ** 2).sum()),
                           "medium": int(((area >= 32 ** 2) & (area < 96 ** 2)).sum()),
                           "large (>=96^2 px)": int((area >= 96 ** 2).sum())},
    }


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

    df, images, sizes, cleaning = load_annotations()
    df, cleaning["dropped_boxes_duplicate_train"] = drop_train_duplicates(df)
    split_images = {
        "train": [n for n in images if camera_of(n) not in VAL_CAMS],
        "val": [n for n in images if camera_of(n) in VAL_CAMS],
    }
    stats = {}
    if args.oversample:  # train only: validation must stay a faithful estimate
        stats["train_before_oversample"] = split_stats(df, split_images["train"])
        split_images["train"], df = oversample(df, split_images["train"], sizes, args.os_thresh, args.os_max_repeat)
    for split, names in split_images.items():
        stats[split] = split_stats(df, names)
        s = stats[split]
        print(f"{split:5s}: {s['images']:4d} images ({s['images_without_boxes']} without boxes), "
              f"{s['boxes']:5d} boxes, cams={s['cameras']}")
        print("       per class:", s["boxes_per_class"])
        print("       per size: ", s["boxes_per_size"])

    write_coco(df, split_images, sizes)
    save_json({"oversample": args.oversample,
               "os_thresh": args.os_thresh if args.oversample else None,
               "os_max_repeat": args.os_max_repeat if args.oversample else None,
               "val_cams": sorted(VAL_CAMS, key=int),
               "cleaning": cleaning,
               "stats": stats}, PREPARE_INFO)
    print(f"COCO dataset -> {COCO_DIR}\nstats -> {PREPARE_INFO}")


if __name__ == "__main__":
    main()
