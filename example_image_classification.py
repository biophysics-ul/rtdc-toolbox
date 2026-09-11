"""Example: image-based RT-DC classification only.

The script recursively finds segmented RTDC files that do not yet have the
specified image-classified analogue, classifies them with the image model, and
prints the classifier metadata.

Quality-control plots are intentionally kept in the separate example script
``example_make_QC_plots_images.py``.

rtdc-toolbox framework version 2.0 (August 2026)
"""

from __future__ import annotations

from pathlib import Path
import multiprocessing as mp

import rtdc_toolbox


def find_unclassified_rtdc_files(
    folder: str | Path,
    ext1: str,
    ext2: str,
) -> list[Path]:
    """Find files ending in ext1 without a corresponding ext2-derived file."""
    folder = Path(folder)
    return sorted(
        path
        for path in folder.rglob(f"*{ext1}.rtdc")
        if not path.with_name(f"{path.stem}{ext2}.rtdc").exists()
    )


def main() -> None:
    """Run image classification on all currently unclassified RTDC files."""

    # =========================================================================
    # USER SETTINGS
    # =========================================================================

    main_folder_with_rtdc = r"Z:\KK\RTDC"
    extension_segmented_rtdc = "_dcn"
    extension_classified_rtdc = "_b7c"

    path_to_model = r"Z:\KK\Training\Models\Blood_7classes_July_2026.pt"
    classifier_name = "Blood_7classes_July_2026"

    overwrite = True

    # =========================================================================
    # FIND FILES
    # =========================================================================

    unclassified_rtdc_list = find_unclassified_rtdc_files(
        folder=main_folder_with_rtdc,
        ext1=extension_segmented_rtdc,
        ext2=extension_classified_rtdc,
    )

    if len(unclassified_rtdc_list) == 0:
        print("No files to process...")
        return

    print(f"Found {len(unclassified_rtdc_list)} RTDC files to process.")

    # =========================================================================
    # CLASSIFICATION
    # =========================================================================

    for i, path_to_rtdc in enumerate(unclassified_rtdc_list, start=1):
        print(
            f"Processing: {path_to_rtdc.relative_to(main_folder_with_rtdc)} "
            f"({i}/{len(unclassified_rtdc_list)})"
        )

        image_rtdc = rtdc_toolbox.classify_rtdc_images_ml(
            path_to_rtdc=path_to_rtdc,
            path_to_model=path_to_model,
            preprocessing_function=rtdc_toolbox.preprocess_rtdc_image,
            classifier_name=classifier_name,
            output_suffix=extension_classified_rtdc,
            overwrite=overwrite,
            verbose=True,
            show_progress=True,
            extra_pixels=15,
            batch_size=2048,
            num_workers=6,
            prefetch_batches=4,
            exclude_x_overlapping_events=True,
            min_x_overlap=0.0,
        )

        print("\nImage-classifier metadata:\n", flush=True)
        rtdc_toolbox.print_rtdc_classifier_metadata(image_rtdc)


if __name__ == "__main__":
    mp.freeze_support()
    main()
