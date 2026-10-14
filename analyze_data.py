from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

from common import CLASS_NAMES, RESULTS_DIR, TEST_IMG_DIR, TRAIN_CSV, TRAIN_IMG_DIR, VAL_CAMS, camera_of

BOX = ["x1", "y1", "x2", "y2"]


def iou_matrix(b):
    x1 = np.maximum(b[:, None, 0], b[None, :, 0])
    y1 = np.maximum(b[:, None, 1], b[None, :, 1])
    x2 = np.minimum(b[:, None, 2], b[None, :, 2])
    y2 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area[:, None] + area[None, :] - inter)


def label_issues(df):
    """Exact duplicates, one box with two classes, and near-identical boxes (IoU > 0.85) with different classes."""
    rows = []
    for name, g in df.groupby("image_id"):
        b, c = g[BOX].to_numpy(float), g.class_id.to_numpy()
        m = iou_matrix(b)
        for i, j in zip(*np.triu_indices(len(b), 1)):
            if m[i, j] <= 0.85 or (c[i] == c[j] and m[i, j] < 1):
                continue
            issue = "duplicate" if c[i] == c[j] else "same_box_two_classes" if m[i, j] == 1 else "near_box_two_classes"
            rows.append({"image_id": name, "issue": issue,
                         "classes": "+".join(CLASS_NAMES[k] for k in sorted((c[i], c[j]))),
                         **dict(zip(BOX, b[i].astype(int)))})
    return pd.DataFrame(rows, columns=["image_id", "issue", "classes", *BOX])


def main():
    df = pd.read_csv(TRAIN_CSV)
    train_imgs = sorted(p.name for p in TRAIN_IMG_DIR.glob("*.jpg"))
    test_imgs = sorted(p.name for p in TEST_IMG_DIR.glob("*.jpg"))
    sizes = Counter()
    for path in [*TRAIN_IMG_DIR.glob("*.jpg"), *TEST_IMG_DIR.glob("*.jpg")]:
        with Image.open(path) as im:
            sizes[im.size] += 1
    df["cam"] = df.image_id.map(camera_of)
    df["split"] = np.where(df.cam.isin(VAL_CAMS), "val", "train")
    df["w"], df["h"] = df.x2 - df.x1, df.y2 - df.y1
    df["side"] = np.sqrt(df.w * df.h)
    train_cams = sorted({camera_of(n) for n in train_imgs}, key=int)
    test_cams = sorted({camera_of(n) for n in test_imgs}, key=int)

    print("=== images")
    print(f"train {len(train_imgs)} ({len(set(train_imgs) - set(df.image_id))} without boxes), test {len(test_imgs)}, "
          f"sizes (w, h): {dict(sizes)}")
    print(f"train cams ({len(train_cams)}): {train_cams}  val cams: {sorted(VAL_CAMS, key=int)}")
    print(f"test cams  ({len(test_cams)}): {test_cams}  shared with train: {sorted(set(train_cams) & set(test_cams))}")
    night = lambda names: np.mean([int(n.split("_")[2][:2]) >= 18 for n in names])
    print(f"frames taken 18:00-21:59 (dusk/night): train {night(train_imgs):.0%}, test {night(test_imgs):.0%}")

    print("\n=== classes")
    car = (df.class_id == 0).sum()
    table = pd.DataFrame({
        "boxes": df.class_id.value_counts(),
        "share": df.class_id.value_counts(normalize=True).map("{:.1%}".format),
        "car/x": (car / df.class_id.value_counts()).round(1),
        "images": df.groupby("class_id").image_id.nunique(),
        "cams": df.groupby("class_id").cam.nunique(),
        "top_cam": df.groupby("class_id").cam.agg(lambda s: f"{s.mode()[0]} ({(s == s.mode()[0]).mean():.0%})"),
        "train": df[df.split == "train"].class_id.value_counts(),
        "val": df[df.split == "val"].class_id.value_counts(),
    }).sort_index().fillna(0)
    table.index = CLASS_NAMES
    print(table.to_string())

    print("\n=== boxes per camera (* = val)")
    per_cam = df.pivot_table(index="cam", columns="class_id", values="x1", aggfunc="count", fill_value=0)
    per_cam = per_cam.reindex(sorted(per_cam.index, key=int)).rename(columns=dict(enumerate(CLASS_NAMES)))
    per_cam.index = [c + ("*" if c in VAL_CAMS else "") for c in per_cam.index]
    print(per_cam.to_string())

    print("\n=== box size (px, image is 352x288)")
    size = df.groupby("class_id").agg(
        w=("w", "median"), h=("h", "median"),
        side_p10=("side", lambda s: s.quantile(0.1)), side_p50=("side", "median"),
        side_p90=("side", lambda s: s.quantile(0.9)),
        small=("side", lambda s: f"{(s < 32).mean():.0%}"),
        under_4px=("x1", lambda s: int(((df.w[s.index] < 4) | (df.h[s.index] < 4)).sum())),
    ).round(1)
    size.index = CLASS_NAMES
    print(size.to_string(), "\n(side = sqrt(area); small = COCO small, area < 32^2)")

    print("\n=== label issues")
    issues = label_issues(df)
    for issue, g in issues.groupby("issue"):
        print(f"{issue:22s} {len(g):4d}  {dict(Counter(g.classes).most_common())}")
    print(f"images without boxes   {len(set(train_imgs) - set(df.image_id)):4d}  (kept as background)")
    out = RESULTS_DIR / "label_issues.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    issues.to_csv(out, index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
