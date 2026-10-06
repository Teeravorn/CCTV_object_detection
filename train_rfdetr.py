"""Fine-tune RF-DETR on the Bangkok CCTV dataset, then evaluate and visualise it.

Follows Roboflow's "How to Train RF-DETR on a Custom Dataset" notebook
(https://github.com/roboflow/rf-detr, roboflow/notebooks how-to-finetune-rf-detr-on-detection-dataset.ipynb),
adapted to rfdetr 1.11:
  1. train          model = RFDETRMedium(); model.train(dataset_dir=..., batch_size * grad_accum_steps = 16)
  2. free GPU       cleanup_gpu_memory(model)
  3. load best      RFDETRMedium(pretrain_weights=".../checkpoint_best_total.pth")
                    + model.inference()   (the notebook's optimize_for_inference(), renamed in rfdetr 1.9+)
  4. evaluate       predict(threshold=0) on every validation image, supervision MeanAveragePrecision,
                    plus the Kaggle-equivalent pycocotools mAP50 (common.coco_evaluate)
  5. visualise      annotation vs detection side by side

Usage:
  python train_rfdetr.py --variant medium --epochs 50 [--aug strong] [--seed 42]
  python train_rfdetr.py --eval-only --weights runs/rfdetr_medium/checkpoint_best_ema.pth
Outputs in runs/rfdetr_<variant>[_os][_aug]/ : checkpoints, eval_valid.json, val_predictions.jpg
"""

import argparse
import gc
import math
import sys
from pathlib import Path
from datetime import datetime

from common import (
    CLASS_NAMES,
    COCO_DIR,
    NUM_CLASSES,
    RUNS_DIR,
    coco_evaluate,
    run_suffix,
    save_json,
)

# rfdetr prints its metric tables with `rich`; the default Windows cp1252 console crashes on them
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8", errors="replace")

VARIANTS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
    "xlarge": "RFDETRXLarge",
    "2xlarge": "RFDETR2XLarge",
}
# checkpoint_best_total.pth is written when training finishes; the others exist while it runs
BEST_CHECKPOINTS = [
    "checkpoint_best_total.pth",
    "checkpoint_best_ema.pth",
    "checkpoint_best_regular.pth",
]
VIS_THRESHOLD = 0.5

# defaults value Parameters
EFFECTIVE_BATCH = 16


def variant_class(variant):
    import rfdetr

    return getattr(rfdetr, VARIANTS[variant])


# ---------------------------------------------------------------- 1. train
def train(args, out_dir: Path):
    from pytorch_lightning import seed_everything

    # Seed before building the model too: model.train(seed=) only seeds once fit starts.
    seed_everything(args.seed, workers=True)
    model = variant_class(args.variant)()
    kwargs = {}
    if args.resolution:
        kwargs["resolution"] = args.resolution
    if args.aug == "strong":
        from augment import rfdetr_aug_config

        kwargs["aug_config"] = rfdetr_aug_config()

    # 'auto': rfdetr probes the batch size and overwrites grad_accum_steps to reach auto_batch_target_effective.
    # Same ceil rule as rfdetr's auto-batch, so e.g. RTX 4060 8 GB with --batch 4 -> 4x4.
    grad_accum = 1 if args.batch == "auto" else math.ceil(EFFECTIVE_BATCH / args.batch)

    # rfdetr's managed schedules: linear warmup from 0 over warmup_epochs, then
    #   cosine: lr * (min_factor + (1 - min_factor) * 0.5 * (1 + cos(pi * progress)))  over the remaining steps
    #   step:   lr until epoch lr_drop, then lr * 0.1
    # Its default is step with lr_drop=100, i.e. a constant lr for any run shorter than 100 epochs.
    if args.lr_schedule == "cosine":
        sched_kwargs = {"min_factor": args.min_lr_factor}
    else:
        sched_kwargs = {"lr_drop": args.lr_drop or round(args.epochs * 2 / 3)}

    model.train(
        dataset_dir=str(COCO_DIR),
        output_dir=str(out_dir),
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=grad_accum,
        auto_batch_target_effective=EFFECTIVE_BATCH,
        lr=args.lr,
        lr_scheduler=args.lr_schedule,
        lr_scheduler_kwargs=sched_kwargs,
        warmup_epochs=args.warmup_epochs,
        num_workers=args.workers,
        early_stopping=True,
        early_stopping_patience=args.patience,
        run_test=False,  # no labelled test split; the real test/ set is scored via predict.py
        tensorboard=True,
        progress_bar="tqdm",
        seed=args.seed,
        **kwargs,
    )
    return model


# ---------------------------------------------------------------- 2. free GPU
def cleanup_gpu_memory(obj=None):
    """Drop the training model before reloading the best checkpoint (notebook's cleanup_gpu_memory)."""
    import torch

    del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(
            f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1024**2:.0f} MB allocated"
        )


# ---------------------------------------------------------------- 3. load best checkpoint
def best_checkpoint(out_dir: Path) -> Path:
    for name in BEST_CHECKPOINTS:
        if (out_dir / name).exists():
            return out_dir / name
    raise FileNotFoundError(f"no {' / '.join(BEST_CHECKPOINTS)} in {out_dir}")


def load_finetuned(variant, weights: Path, compile_model=True):
    model = variant_class(variant)(
        pretrain_weights=str(weights), num_classes=NUM_CLASSES
    )
    if compile_model:
        model.inference(
            compile=True, batch_size=1
        )  # = optimize_for_inference() in the notebook
    return model


def predict(model, image, threshold):
    """model.predict, minus the DETR "no object" slot (class_id == NUM_CLASSES) and zero-size boxes."""
    d = model.predict(image, threshold=threshold, include_source_image=False)
    w, h = d.xyxy[:, 2] - d.xyxy[:, 0], d.xyxy[:, 3] - d.xyxy[:, 1]
    return d[(d.class_id < NUM_CLASSES) & (w >= 1) & (h >= 1)]


# ---------------------------------------------------------------- 4. evaluate
def evaluate(model, out_dir: Path):
    import supervision as sv
    from PIL import Image
    from supervision.metrics import MeanAveragePrecision
    from tqdm import tqdm

    ds = sv.DetectionDataset.from_coco(
        images_directory_path=str(COCO_DIR / "valid"),
        annotations_path=str(COCO_DIR / "valid" / "_annotations.coco.json"),
    )
    targets, predictions, kaggle_preds = [], [], {}
    for path, _, annotations in tqdm(ds, desc="valid"):
        detections = predict(
            model, Image.open(path).convert("RGB"), threshold=0.0
        )  # notebook: threshold=0
        targets.append(annotations)
        predictions.append(detections)
        kaggle_preds[Path(path).name] = [
            (int(c), float(s), *map(float, xy))
            for c, s, xy in zip(
                detections.class_id, detections.confidence, detections.xyxy
            )
        ]

    sv_map = MeanAveragePrecision().update(predictions, targets).compute()
    kaggle = coco_evaluate(
        kaggle_preds
    )  # pycocotools, IoU 0.5, maxDets 100 = the leaderboard metric

    per_class = {
        CLASS_NAMES[int(c)]: float(ap[0])
        for c, ap in zip(sv_map.matched_classes, sv_map.ap_per_class)
    }
    result = {
        "kaggle_mAP50": kaggle["mAP50"],
        "kaggle_mAP50-95": kaggle["mAP50-95"],
        "kaggle_per_class_AP50": {
            k: v["mAP50"] for k, v in kaggle["per_class"].items()
        },
        "supervision_mAP50": float(sv_map.mAP_scores[0]),
        "supervision_mAP50-95": float(sv_map.mAP_scores.mean()),
        "supervision_per_class_AP50": per_class,
    }
    print(
        f"\nvalid mAP50 (Kaggle metric) = {result['kaggle_mAP50']:.4f}   "
        f"supervision mAP50 = {result['supervision_mAP50']:.4f}   mAP50-95 = {result['kaggle_mAP50-95']:.4f}"
    )
    for name, ap in result["kaggle_per_class_AP50"].items():
        print(f"  {name:11s} AP50 = {ap:.4f}")
    save_json(result, out_dir / "eval_valid.json")
    return ds


# ---------------------------------------------------------------- 5. visualise
def visualize(model, ds, out_dir: Path, n: int):
    import numpy as np
    import supervision as sv
    from PIL import Image

    palette = sv.ColorPalette.from_hex(
        [
            "#ffff00",
            "#ff9b00",
            "#ff66ff",
            "#3399ff",
            "#ff66b2",
            "#ff8080",
            "#b266ff",
            "#66ff66",
        ]
    )
    step = max(len(ds) // n, 1)
    rows = []
    for i in range(0, step * n, step):
        path, _, annotations = ds[i]
        image = Image.open(path).convert("RGB")
        detections = predict(model, image, threshold=VIS_THRESHOLD)
        box = sv.BoxAnnotator(color=palette, thickness=1)
        label = sv.LabelAnnotator(
            color=palette, text_color=sv.Color.BLACK, text_scale=0.3, text_padding=2
        )
        gt_img = label.annotate(
            box.annotate(image.copy(), annotations),
            annotations,
            [CLASS_NAMES[c] for c in annotations.class_id],
        )
        dt_img = label.annotate(
            box.annotate(image.copy(), detections),
            detections,
            [
                f"{CLASS_NAMES[c]} {s:.2f}"
                for c, s in zip(detections.class_id, detections.confidence)
            ],
        )
        rows.append(np.hstack([np.asarray(gt_img), np.asarray(dt_img)]))
    path = out_dir / "val_predictions.jpg"
    Image.fromarray(np.vstack(rows)).save(path, quality=90)
    print(f"annotation (left) vs detection conf>={VIS_THRESHOLD} (right) -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="medium", choices=VARIANTS)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch",type=lambda s: s if s == "auto" else int(s),default=4,
                    help=f"int, or 'auto' = rfdetr probes the largest batch that fits",)
    ap.add_argument("--lr", type=float, default=1e-4, help="initial learning rate")
    ap.add_argument("--lr-schedule",choices=["cosine", "step"],default="cosine",
                    help="cosine: anneal to lr * --min-lr-factor by the last epoch; step: lr * 0.1 after --lr-drop",
    )
    ap.add_argument("--warmup-epochs", type=float, default=1.0, help="linear warmup from 0 to --lr")
    ap.add_argument("--min-lr-factor",type=float,default=0.01,
                    help="cosine only: final lr = lr * this",)
    ap.add_argument("--lr-drop",type=int,default=None,
                    help="step only: epoch of the 10x drop (default 2/3 of --epochs)",)
    ap.add_argument("--resolution",type=int,default=None,
                    help="default: model's native (nano 384, small 512, medium 576, large 704); must be /32",)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed",type=int,default=42,
                    help="fixed for reproducibility; GPU kernels are not bit-exact, so reruns are close, not identical",)
    ap.add_argument("--aug",choices=["default", "strong"],default="default",
                    help="strong: HorizontalFlip + photometric (fog/blur/noise/JPEG/...), see augment.py",)
    ap.add_argument("--eval-only",action="store_true",help="skip training; evaluate --weights or the run's best",)
    ap.add_argument("--weights", type=Path, default=None, help="checkpoint for --eval-only")
    ap.add_argument("--no-compile", action="store_true", help="skip model.inference() compilation")
    ap.add_argument("--vis", type=int, default=6, help="validation images to draw (0 = none)")
    args = ap.parse_args()

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = RUNS_DIR / f"rfdetr_{args.variant}{run_suffix(args.aug)}_{datetime_str}"
    if not args.eval_only:
        cleanup_gpu_memory(train(args, out_dir))

    weights = args.weights or best_checkpoint(out_dir)
    out_dir = weights.parent
    print(f"loading {weights}")
    model = load_finetuned(args.variant, weights, compile_model=not args.no_compile)
    ds = evaluate(model, out_dir)
    if args.vis:
        visualize(model, ds, out_dir, args.vis)


if __name__ == "__main__":
    main()
