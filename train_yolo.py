"""Fine-tune YOLO26 (COCO-pretrained) on the traffic dataset.

Usage:  python train_yolo.py --model yolo26s.pt --epochs 100 --imgsz 640
Best weights -> runs/yolo26s/weights/best.pt
"""
import argparse

from common import RUNS_DIR, YOLO_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26m.pt", help="yolo26n/s/m/l/x.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640, help="images are 352x288; upscaling helps small objects")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(YOLO_DIR / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        project=str(RUNS_DIR),
        name=args.name or args.model.replace(".pt", ""),
        exist_ok=True,
        seed=0,
        cos_lr=True,
        close_mosaic=10,
        plots=True,
    )


if __name__ == "__main__":
    main()
