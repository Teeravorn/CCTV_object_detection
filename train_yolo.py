"""Fine-tune YOLO26 (COCO-pretrained) on the traffic dataset.

Usage:  python train_yolo.py --model yolo26m.pt --epochs 100 --imgsz 640 [--aug strong]
Best weights -> runs/yolo26m[_os][_aug]/weights/best.pt
(_os when data/ was built with prepare_data.py --oversample, _aug with --aug strong, _pw<x> with --cls-pw x)
"""
import argparse

from common import RUNS_DIR, YOLO_DIR, run_suffix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26m.pt", help="yolo26n/s/m/l/x.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640, help="images are 352x288; upscaling helps small objects")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default=None)
    ap.add_argument("--aug", choices=["default", "strong"], default="default",
                    help="strong: extra photometric (fog/blur/noise/JPEG/...) + mild geometric, see augment.py")
    ap.add_argument("--cls-pw", type=float, default=0.0,
                    help="class-weighted cls loss: weight = (1/count)^cls_pw, mean 1 (0=off, try 0.3)")
    args = ap.parse_args()

    from ultralytics import YOLO

    aug_kwargs = {}
    if args.aug == "strong":
        from augment import YOLO_STRONG_HYP, yolo_albumentations
        aug_kwargs = {**YOLO_STRONG_HYP, "augmentations": yolo_albumentations()}

    model = YOLO(args.model)
    model.train(
        data=str(YOLO_DIR / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        project=str(RUNS_DIR),
        name=args.name or args.model.replace(".pt", "") + run_suffix(args.aug) + (f"_pw{args.cls_pw:g}" if args.cls_pw else ""),
        exist_ok=True,
        seed=0,
        cos_lr=True,
        close_mosaic=10,
        plots=True,
        cls_pw=args.cls_pw,
        **aug_kwargs,
    )


if __name__ == "__main__":
    main()
