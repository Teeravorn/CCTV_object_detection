"""Fine-tune RF-DETR (COCO-pretrained) on the traffic dataset.

Usage:  python train_rfdetr.py --variant medium --epochs 50
Best weights -> runs/rfdetr_medium/checkpoint_best_total.pth
"""
import argparse
import sys

from common import COCO_DIR, RUNS_DIR

# rfdetr prints its metric tables with `rich`; the default Windows cp1252 console crashes on them
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8", errors="replace")

VARIANTS = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "large": "RFDETRLarge"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="medium", choices=VARIANTS)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4, help="effective batch = batch * grad_accum")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resolution", type=int, default=None,
                    help="default: model's native (nano 384, small 512, medium 576, large 704); must be /32")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    import rfdetr

    model = getattr(rfdetr, VARIANTS[args.variant])()
    kwargs = {}
    if args.resolution:
        kwargs["resolution"] = args.resolution
    model.train(
        dataset_dir=str(COCO_DIR),
        output_dir=str(RUNS_DIR / f"rfdetr_{args.variant}"),
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        num_workers=args.workers,
        early_stopping=True,
        early_stopping_patience=args.patience,
        tensorboard=True,
        progress_bar="tqdm",
        **kwargs,
    )


if __name__ == "__main__":
    main()
