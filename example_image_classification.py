"""Example: image-based RT-DC classification and reports.

The script recursively finds segmented RTDC files that do not yet have the
specified image-classified analogue, classifies them with the image model, and
exports metadata and quality-control reports.

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
    """Run image classification and reports on all unclassified RTDC files."""
    main_folder_with_rtdc = r"Z:\KK_Staro\Brez"
    extension_segmented_rtdc = "_dcn"
    extension_classified_rtdc = "_b7c"

    unclassified_rtdc_list = find_unclassified_rtdc_files(
        folder=main_folder_with_rtdc,
        ext1=extension_segmented_rtdc,
        ext2=extension_classified_rtdc,
    )

    if len(unclassified_rtdc_list) == 0:
        print("No files to process...")
        return

    print(f"Found {len(unclassified_rtdc_list)} RTDC files to process.")

    for i, path_to_rtdc in enumerate(unclassified_rtdc_list, start=1):
        print(
            f"Processing: {path_to_rtdc.relative_to(main_folder_with_rtdc)} "
            f"({i}/{len(unclassified_rtdc_list)})"
        )

        image_rtdc = rtdc_toolbox.classify_rtdc_images_ml(
            path_to_rtdc=path_to_rtdc,
            path_to_model=r"D:\RTDC\Training\Models\Blood_7classes_July_2026.pt",
            preprocessing_function=rtdc_toolbox.preprocess_rtdc_image,
            classifier_name="Blood_7classes_July_2026",
            output_suffix=extension_classified_rtdc,
            overwrite=True,
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

        rtdc_toolbox.export_rtdc_class_stability_report(
            path_to_rtdc=image_rtdc,
            classifier_name="default",
            features_to_plot=("area_um", "deform", "bright_avg", "area_ratio"),
            n_window=10_000,
            step=1_000,
        )

        rtdc_toolbox.export_rtdc_class_examples(
            path_to_rtdc=image_rtdc,
            classifier_name="default",
            report_ext="_classes",
            N_img=40,
            preprocessing_function=rtdc_toolbox.preprocess_rtdc_image,
            randomize=True,
            prob_thresh=0.95,
            keep_above_thresh=True,
            n_rows=4,
        )


if __name__ == "__main__":
    mp.freeze_support()
    main()
