"""Find duplicate labels in train.csv (one object labelled with two classes) and render them for manual review.

A pair is a duplicate when both boxes are in the same image, have different classes and IoU > --iou
(same-class duplicates are not reported). For every pair it writes
one review image: the full frame (both boxes highlighted, other boxes in grey) next to an upscaled crop.
The `dedupe_action` column shows what the old `prepare_data.py --dedupe` rule would have done with the pair
(that flag has been removed; nothing applies these decisions to the training data any more).

Review: write the class to keep (or `both`) in the `review` column of duplicates.csv. On the next run the
filled-in cells are first saved to results/duplicate_overrides.csv, so re-running never loses a review;
the column is then pre-filled from that file.

Usage:  python find_duplicates.py                  # IoU > 0.8 -> results/duplicates/
        python find_duplicates.py --iou 0.5 --out results/dup_05

Output:
  <out>/duplicates.csv            one row per pair (+ empty `review` column to fill in)
  <out>/<ClassA+ClassB>/*.jpg     one review image per pair, grouped by class pair
  <out>/overview_<pair>.jpg       grid of all crops of that class pair, for quick browsing
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from common import CLASS_NAMES, RESULTS_DIR, TRAIN_CSV, TRAIN_IMG_DIR, VAL_CAMS, camera_of

# The dedupe rules and helpers below used to live in prepare_data.py / common.py (prepare_data.py --dedupe),
# which has since been removed; they are kept here so this review tool still runs on its own.
DEDUPE_OVERRIDES = RESULTS_DIR / "duplicate_overrides.csv"
# Same object labelled twice with different classes (IoU > DEDUPE_IOU): the lower-ranked class was dropped.
# Bus+Truck / Bus+Car were genuinely mixed, so both boxes were kept.
DEDUPE_IOU = 0.85
DEDUPE_RANK = {"Truck": 0, "Car": 1, "Bus": 2, "Motorcycle": 3, "Tuktuk": 3, "Van": 3, "Pickup": 3, "Songthaew": 3}
DEDUPE_AMBIGUOUS = [{"Bus", "Truck"}, {"Bus", "Car"}]


def _iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def load_overrides(path=DEDUPE_OVERRIDES):
    """Manual review decisions: {frozenset((annotation id, annotation id)): class name to keep, or "both"}."""
    if not path.exists():
        return {}
    ov = pd.read_csv(path, encoding="utf-8-sig", dtype={"review": str}, keep_default_na=False)
    valid = {n.lower(): n for n in CLASS_NAMES} | {"both": "both"}
    out = {}
    for r in ov.itertuples():
        keep = valid.get(r.review.strip().lower())
        if keep is None:
            raise ValueError(f"{path.name}: review {r.review!r} for ids {r.id_a}/{r.id_b} must be a class name or 'both'")
        out[frozenset((int(r.id_a), int(r.id_b)))] = keep
    return out

FULL_SCALE = 2        # 352x288 frames are tiny; draw on a 2x copy
CROP_SIZE = 288       # crop is upscaled so its longer side is this many px (= height of the full view)
THUMB = 128           # crop size in the overview grids
COLOR_A, COLOR_B, COLOR_OTHER = (255, 40, 40), (40, 220, 255), (150, 150, 150)


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def find_pairs(df, iou_thr):
    rows = []
    for image_id, g in df.groupby("image_id"):
        recs = list(g.itertuples())
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = recs[i], recs[j]
                if a.class_id == b.class_id:  # only cross-class double labels are reported
                    continue
                iou = _iou((a.x1, a.y1, a.x2, a.y2), (b.x1, b.y1, b.x2, b.y2))
                if iou > iou_thr:
                    rows.append({"image_id": image_id, "a": a, "b": b, "iou": iou})
    return rows


OVERRIDE_COLS = ["image_id", "id_a", "class_a", "id_b", "class_b", "kind", "review"]


def save_reviews(csv_path):
    """Merge filled-in `review` cells of a previous duplicates.csv into DEDUPE_OVERRIDES (newest wins)."""
    if not csv_path.exists():
        return
    new = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    new = new[new.review.str.strip() != ""]
    if new.empty:
        return
    old = (pd.read_csv(DEDUPE_OVERRIDES, encoding="utf-8-sig", dtype=str, keep_default_na=False)
           if DEDUPE_OVERRIDES.exists() else pd.DataFrame(columns=OVERRIDE_COLS))
    merged = pd.concat([old[OVERRIDE_COLS], new[OVERRIDE_COLS]], ignore_index=True)
    merged["_key"] = [frozenset((int(a), int(b))) for a, b in zip(merged.id_a, merged.id_b)]
    merged = merged.drop_duplicates("_key", keep="last").drop(columns="_key")
    merged.to_csv(DEDUPE_OVERRIDES, index=False, encoding="utf-8-sig")
    load_overrides()  # fail now on a typo in a review cell
    print(f"saved {len(new)} reviews -> {DEDUPE_OVERRIDES} ({len(merged)} total)")


def dedupe_action(name_a, name_b, iou, review=None):
    """Same rules as prepare_data.dedupe() (which also skips boxes already dropped by an earlier pair)."""
    if review:
        return f"manual: keep {review}"
    if iou <= DEDUPE_IOU:
        return f"keep both (IoU <= {DEDUPE_IOU})"
    if {name_a, name_b} in DEDUPE_AMBIGUOUS or DEDUPE_RANK[name_a] == DEDUPE_RANK[name_b]:
        return "keep both (ambiguous)"
    return f"drop {name_a}" if DEDUPE_RANK[name_a] < DEDUPE_RANK[name_b] else f"drop {name_b}"


def draw_box(d, box, color, width, text=None, below=False, fnt=None):
    x1, y1, x2, y2 = box
    d.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if text:
        tw, th = d.textbbox((0, 0), text, font=fnt)[2:]
        ty = y2 + 1 if below else y1 - th - 3
        d.rectangle([x1, ty, x1 + tw + 4, ty + th + 2], fill=color)
        d.text((x1 + 2, ty), text, fill="black", font=fnt)


def crop_box(a, b, W, H):
    """Union of both boxes plus context padding, clipped to the image."""
    x1, y1 = min(a.x1, b.x1), min(a.y1, b.y1)
    x2, y2 = max(a.x2, b.x2), max(a.y2, b.y2)
    pad = max(8, 0.5 * max(x2 - x1, y2 - y1))
    return max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)


def render(pair, all_boxes, k):
    a, b = pair["a"], pair["b"]
    na, nb = CLASS_NAMES[a.class_id], CLASS_NAMES[b.class_id]
    img = Image.open(TRAIN_IMG_DIR / pair["image_id"]).convert("RGB")
    W, H = img.size

    # full frame, 2x
    full = img.resize((W * FULL_SCALE, H * FULL_SCALE), Image.BILINEAR)
    d = ImageDraw.Draw(full)
    s = FULL_SCALE
    for r in all_boxes.itertuples():
        if r.id not in (a.id, b.id):
            d.rectangle([r.x1 * s, r.y1 * s, r.x2 * s, r.y2 * s], outline=COLOR_OTHER, width=1)
    f_small = font(12)
    draw_box(d, (a.x1 * s, a.y1 * s, a.x2 * s, a.y2 * s), COLOR_A, 2, f"A {na}", below=False, fnt=f_small)
    draw_box(d, (b.x1 * s, b.y1 * s, b.x2 * s, b.y2 * s), COLOR_B, 2, f"B {nb}", below=True, fnt=f_small)

    # crop, upscaled; box A drawn outside box B so exact-duplicate boxes stay distinguishable
    cx1, cy1, cx2, cy2 = crop_box(a, b, W, H)
    crop = img.crop((int(cx1), int(cy1), int(round(cx2)), int(round(cy2))))
    z = CROP_SIZE / max(crop.size)
    crop = crop.resize((max(1, round(crop.width * z)), max(1, round(crop.height * z))), Image.LANCZOS)
    dc = ImageDraw.Draw(crop)
    ox, oy = int(cx1), int(cy1)
    to_crop = lambda r: ((r.x1 - ox) * z, (r.y1 - oy) * z, (r.x2 - ox) * z, (r.y2 - oy) * z)
    ba, bb = to_crop(a), to_crop(b)
    dc.rectangle([ba[0] - 2, ba[1] - 2, ba[2] + 2, ba[3] + 2], outline=COLOR_A, width=2)
    dc.rectangle(list(bb), outline=COLOR_B, width=2)

    # panel: [full | crop] + caption bar
    f_cap = font(16)
    cap_h = 46
    panel = Image.new("RGB", (full.width + CROP_SIZE + 10, full.height + cap_h), (25, 25, 25))
    panel.paste(full, (0, cap_h))
    panel.paste(crop, (full.width + 10, cap_h))
    dp = ImageDraw.Draw(panel)
    split = "VAL" if camera_of(pair["image_id"]) in VAL_CAMS else "train"
    dp.text((6, 4), f"#{k}  {pair['image_id']}  [{split}]   IoU={pair['iou']:.3f}", fill="white", font=f_cap)
    dp.text((6, 24), f"A: id={a.id} {na}", fill=COLOR_A, font=f_cap)
    dp.text((220, 24), f"B: id={b.id} {nb}", fill=COLOR_B, font=f_cap)
    dp.text((440, 24), f"--dedupe: {pair['action']}", fill=(255, 220, 80), font=f_cap)
    return panel, crop


def overview(thumbs, path, cols=10):
    """thumbs: [(crop image, caption)] -> one grid image."""
    f = font(11)
    cell_h = THUMB + 14
    grid = Image.new("RGB", (cols * THUMB, cell_h * ((len(thumbs) + cols - 1) // cols)), (0, 0, 0))
    d = ImageDraw.Draw(grid)
    for i, (im, cap) in enumerate(thumbs):
        x, y = (i % cols) * THUMB, (i // cols) * cell_h
        t = im.copy()
        t.thumbnail((THUMB, THUMB))
        grid.paste(t, (x + (THUMB - t.width) // 2, y + 14))
        d.text((x + 2, y + 1), cap, fill="yellow", font=f)
    grid.save(path, quality=92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iou", type=float, default=0.8, help="pairs with IoU above this are reported")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "duplicates")
    ap.add_argument("--no-images", action="store_true", help="only write duplicates.csv")
    args = ap.parse_args()

    save_reviews(args.out / "duplicates.csv")  # before the output dir is wiped
    overrides = load_overrides()

    df = pd.read_csv(TRAIN_CSV)
    pairs = find_pairs(df, args.iou)
    for p in pairs:
        na, nb = CLASS_NAMES[p["a"].class_id], CLASS_NAMES[p["b"].class_id]
        p["kind"] = "+".join(sorted((na, nb)))
        p["review"] = overrides.get(frozenset((p["a"].id, p["b"].id)), "")
        p["action"] = dedupe_action(na, nb, p["iou"], p["review"])
    pairs.sort(key=lambda p: (p["kind"], p["image_id"]))

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    rows = []
    for k, p in enumerate(pairs):
        a, b = p["a"], p["b"]
        file = f"{p['kind']}/{k:04d}_{Path(p['image_id']).stem}_{a.id}-{b.id}.jpg"
        rows.append({
            "pair": k, "image_id": p["image_id"], "camera": camera_of(p["image_id"]),
            "split": "val" if camera_of(p["image_id"]) in VAL_CAMS else "train",
            "kind": p["kind"], "iou": round(p["iou"], 4),
            "id_a": a.id, "class_a": CLASS_NAMES[a.class_id], "box_a": f"{a.x1},{a.y1},{a.x2},{a.y2}",
            "id_b": b.id, "class_b": CLASS_NAMES[b.class_id], "box_b": f"{b.x1},{b.y1},{b.x2},{b.y2}",
            "dedupe_action": p["action"], "review_image": file, "review": p["review"],
        })
    report = pd.DataFrame(rows)
    report.to_csv(args.out / "duplicates.csv", index=False, encoding="utf-8-sig")  # utf-8-sig: opens in Excel

    print(f"{len(pairs)} duplicate pairs (IoU > {args.iou}) in {report.image_id.nunique() if rows else 0} images")
    if rows:
        summary = report.groupby("kind").agg(pairs=("pair", "size"), val=("split", lambda s: (s == "val").sum()),
                                             mean_iou=("iou", "mean")).sort_values("pairs", ascending=False)
        print(summary.round(3).to_string())
        print("\n--dedupe would:", report.dedupe_action.value_counts().to_dict())

    if not args.no_images and rows:
        by_img = {k: g for k, g in df.groupby("image_id")}
        thumbs = {}
        for k, p in enumerate(pairs):
            panel, crop = render(p, by_img[p["image_id"]], k)
            path = args.out / rows[k]["review_image"]
            path.parent.mkdir(exist_ok=True)
            panel.save(path, quality=92)
            thumbs.setdefault(p["kind"], []).append((crop, f"#{k} {p['iou']:.2f}"))
        for kind, t in thumbs.items():
            overview(t, args.out / f"overview_{kind}.jpg")
    print(f"\n-> {args.out}/duplicates.csv" + ("" if args.no_images else " + review images per class pair"))


if __name__ == "__main__":
    main()
