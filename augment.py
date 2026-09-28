"""Augmentation presets shared by train_yolo.py and train_rfdetr.py (--aug strong).

Test cameras are unseen and include hazy/foggy lenses, dusk/night shots and heavy JPEG artefacts,
so the extra augmentations are mostly photometric. Geometric ones stay mild because objects are tiny
(~17x27 px): rotation/shear loosen boxes and strong down-scaling makes objects vanish.

Pixel-level transforms need `albumentations` (<3 for rfdetr):
  pip install --only-binary=:all: "albumentations>=1.4.24,<3"
"""

# Photometric pipeline as Albumentations specs [{name: params}]; built for YOLO, passed as-is to RF-DETR.
PIXEL_SPECS = [
    {"RandomBrightnessContrast": {"brightness_limit": 0.25, "contrast_limit": 0.25, "p": 0.4}},
    {"RandomGamma": {"gamma_limit": (70, 130), "p": 0.2}},
    {"RandomFog": {"fog_coef_range": (0.1, 0.3), "alpha_coef": 0.08, "p": 0.1}},
    {"OneOf": {"p": 0.15, "transforms": [
        {"GaussianBlur": {"blur_limit": (3, 5), "p": 1.0}},
        {"MotionBlur": {"blur_limit": (3, 7), "p": 1.0}},
    ]}},
    {"GaussNoise": {"std_range": (0.02, 0.08), "p": 0.1}},
    {"ImageCompression": {"quality_range": (30, 80), "p": 0.2}},
    {"CLAHE": {"clip_limit": 2.0, "p": 0.1}},
    {"ToGray": {"p": 0.05}},
]

# Ultralytics hyperparameters for --aug strong (anything not listed keeps the Ultralytics default:
# fliplr=0.5, flipud=0, translate=0.1, scale=0.5, shear=0, perspective=0, mosaic=1.0, hsv_h=0.015, hsv_s=0.7)
YOLO_STRONG_HYP = {
    "hsv_v": 0.5,
    "degrees": 3.0,
    "mixup": 0.1,
}


def _build(name, params):
    import albumentations as A

    params = dict(params)
    if "transforms" in params:
        params["transforms"] = [_build(*next(iter(t.items()))) for t in params["transforms"]]
    return getattr(A, name)(**params)


def yolo_albumentations():
    """List of Albumentations transforms for Ultralytics' `augmentations=` train argument."""
    return [_build(*next(iter(spec.items()))) for spec in PIXEL_SPECS]


def rfdetr_aug_config():
    """RF-DETR `aug_config`: its default is just HorizontalFlip, which a custom config replaces, so keep it."""
    cfg = {"HorizontalFlip": {"p": 0.5}}
    for spec in PIXEL_SPECS:
        cfg.update(spec)
    return cfg
