"""Augmentation preset for train_rfdetr.py --aug strong.

Test cameras are unseen and include hazy/foggy lenses, dusk/night shots and heavy JPEG artefacts,
so the extra augmentations are mostly photometric. Geometric ones stay mild because objects are tiny
(~17x27 px): rotation/shear loosen boxes and strong down-scaling makes objects vanish.

Pixel-level transforms need `albumentations` (<3 for rfdetr):
  pip install --only-binary=:all: "albumentations>=1.4.24,<3"
"""

# Photometric pipeline as Albumentations specs [{name: params}], passed as-is to RF-DETR.
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

def rfdetr_aug_config():
    """RF-DETR `aug_config`: its default is just HorizontalFlip, which a custom config replaces, so keep it."""
    cfg = {"HorizontalFlip": {"p": 0.5}}
    for spec in PIXEL_SPECS:
        cfg.update(spec)
    return cfg
