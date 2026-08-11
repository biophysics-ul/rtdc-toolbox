
"""Example: train an RT-DC classifier from preprocessed grayscale images.

rtdc-toolbox framework version 2.0 (August 2026)

The example reads a ZIP archive with one top-level folder per class::

    RBC/
        img001.png
    platelets/
        img010.png
    WBC/
        img020.png

The trained model is image-only.  Future feature-based or multimodal training
functions can therefore use different names without ambiguity.
"""

from __future__ import annotations

from pathlib import Path

import rtdc_toolbox


# =============================================================================
# USER SETTINGS
# =============================================================================

IMAGE_CNN_BACKBONES = ("convnext_tiny",)
# Other supported choices: "resnet18", "resnet50", "efficientnet_b0".

TRAINING_IMAGE_ARCHIVE = Path(r"D:\RTDC\Training\TrainingSet.ZIP")
TRAINING_RESULTS_DIR = Path(r"D:\RTDC\Training\TrainingResults")


# =============================================================================
# TRAINING
# =============================================================================


def main() -> None:
    """Train one image classifier for each configured CNN backbone."""
    if not TRAINING_IMAGE_ARCHIVE.exists():
        raise FileNotFoundError(
            f"Training image archive not found: {TRAINING_IMAGE_ARCHIVE}"
        )

    print("Starting image-model training from ZIP archive...", flush=True)

    for cnn_backbone in IMAGE_CNN_BACKBONES:
        output_dir = TRAINING_RESULTS_DIR / cnn_backbone
        model_path = rtdc_toolbox.train_model_on_images(
            # Image data and output location.
            path_to_training_set=str(TRAINING_IMAGE_ARCHIVE),
            training_results_dir=str(output_dir),
            model_name=cnn_backbone,
            zip=True,
            img_ext=".png",

            # Image-CNN architecture.
            cnn_backbone=cnn_backbone,
            pretrained=True,

            # Training schedule.
            n_epochs=10,
            W0=160,
            val_split=0.2,

            # Performance.
            batch_size=256,
            cache_in_ram=True,
            # On Windows, use zero workers with a RAM cache to avoid copying the
            # entire cached image set into every worker process.
            num_workers=6,

            # Optimizer.
            lr=3e-4,
            weight_decay=1e-4,

            # Image augmentation.
            brightness_jitter=0.02,
            contrast_jitter=0.05,

            # Reproducibility.
            seed=123,
        )
        print(f"Saved image model: {model_path}", flush=True)


if __name__ == "__main__":
    main()
