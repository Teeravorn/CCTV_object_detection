"""RF-DETR inference wrapper shared by evaluate.py and predict.py.

The detector returns {file_name: [(class_id, score, x1, y1, x2, y2), ...]} in original-image pixels.
"""
import time
from pathlib import Path

from common import NUM_CLASSES

# Keep low-confidence boxes: mAP needs the full precision/recall curve.
EVAL_CONF = 0.001
MAX_DET = 300
# Boxes predicted (partly) outside the image get clamped to the border and can collapse to zero
# width/height (e.g. y1 == y2 == 288). Drop anything thinner than this many pixels.
MIN_SIZE = 1.0


def _keep(xyxy) -> bool:
    x1, y1, x2, y2 = xyxy
    return x2 - x1 >= MIN_SIZE and y2 - y1 >= MIN_SIZE


class RFDETRDetector:
    name = "RF-DETR"

    def __init__(self, checkpoint: Path):
        from rfdetr import RFDETR
        self.model = RFDETR.from_checkpoint(str(checkpoint), trust_checkpoint=True)

    def predict(self, paths: list[Path], batch: int = 16) -> dict:
        from PIL import Image
        out = {}
        for i in range(0, len(paths), batch):
            chunk = paths[i:i + batch]
            imgs = [Image.open(p).convert("RGB") for p in chunk]
            dets = self.model.predict(imgs, threshold=EVAL_CONF, include_source_image=False)
            if not isinstance(dets, list):
                dets = [dets]
            for p, d in zip(chunk, dets):
                # The head has num_classes + 1 logits; the last one is the DETR "no object" slot, which
                # top-k can still pick at very low thresholds (class_id == NUM_CLASSES). Drop it.
                out[Path(p).name] = [(int(c), float(s), *map(float, xy))
                                     for c, s, xy in zip(d.class_id, d.confidence, d.xyxy)
                                     if c < NUM_CLASSES and _keep(xy)]
        return out


def timed_predict(detector, paths: list[Path]) -> tuple[dict, float]:
    """Run prediction and return (predictions, ms per image). First 8 images warm up the GPU."""
    import torch
    detector.predict(paths[:8])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    preds = detector.predict(paths)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return preds, (time.perf_counter() - t0) * 1000 / max(len(paths), 1)
