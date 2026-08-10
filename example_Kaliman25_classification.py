"""Example: Kaliman25 WBC classification and reports.

Set USE_IMAGE_SELECTED_WBCS to choose between two workflows:

False:
    Run the original Kaliman25 workflow on whole blood. Kaliman Step 1 performs
    the crude WBC preselection.

True:
    Read WBCs from the corresponding image-classified RTDC file and pass that
    mask to Kaliman25 with preselected_wbcs=True. Kaliman Step 1 is skipped and
    the image-selected WBCs enter directly at Step 2.

Framework API: 2.3.2-unified-classifier-api
"""

from __future__ import annotations

from pathlib import Path

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


# =============================================================================
# USER SETTINGS
# =============================================================================

main_folder_with_rtdc = r"Z:\KK_Staro\Brez"

extension_segmented_rtdc = "_dcn"
extension_image_classified_rtdc = "_b7c"
extension_kaliman_rtdc = "_kaliman25a"

USE_IMAGE_SELECTED_WBCS = True
WBC_PROBABILITY_THRESHOLD = 0.90


# =============================================================================
# CLASSIFICATION
# =============================================================================

unclassified_rtdc_list = find_unclassified_rtdc_files(
    folder=main_folder_with_rtdc,
    ext1=extension_segmented_rtdc,
    ext2=extension_kaliman_rtdc,
)

print(f"Found {len(unclassified_rtdc_list)} RTDC files to process.")

for i, path_to_rtdc in enumerate(unclassified_rtdc_list, start=1):
    print(
        f"Processing: {path_to_rtdc.relative_to(main_folder_with_rtdc)} "
        f"({i}/{len(unclassified_rtdc_list)})"
    )

    if USE_IMAGE_SELECTED_WBCS:
        image_rtdc = path_to_rtdc.with_name(
            f"{path_to_rtdc.stem}{extension_image_classified_rtdc}.rtdc"
        )

        if not image_rtdc.exists():
            print(f"  Skipping: image-classified file not found: {image_rtdc}")
            continue

        wbc_mask = rtdc_toolbox.make_classifier_mask(
            path_or_ds=image_rtdc,
            classifier_name="default",
            classes="WBC",
            probability_threshold=WBC_PROBABILITY_THRESHOLD,
            probability_comparison=">",
        )

        print(f"WBC mask selected {int(wbc_mask.sum()):,} events.", flush=True)

        preselected_wbcs = True
        mask_description = (
            f"WBC events selected from {image_rtdc.name} with "
            f"probability > {WBC_PROBABILITY_THRESHOLD:g}"
        )

    else:
        wbc_mask = None
        preselected_wbcs = False
        mask_description = None

    kaliman_rtdc = rtdc_toolbox.classify_rtdc_WBC_Kaliman25(
        path_to_rtdc=path_to_rtdc,
        classifier_name="Kaliman25_WBC",
        mask=wbc_mask,
        mask_description=mask_description,
        preselected_wbcs=preselected_wbcs,
        output_suffix=extension_kaliman_rtdc,
        overwrite=True,
        verbose=True,
    )

    print("\nKaliman25-classifier metadata:\n", flush=True)
    rtdc_toolbox.print_rtdc_classifier_metadata(kaliman_rtdc)

    rtdc_toolbox.export_rtdc_class_stability_report(
        path_to_rtdc=kaliman_rtdc,
        classifier_name="default",
        features_to_plot=("area_um", "deform", "bright_avg", "area_ratio"),
        n_window=10_000,
        step=1_000,
    )

    rtdc_toolbox.export_rtdc_class_examples(
        path_to_rtdc=kaliman_rtdc,
        classifier_name="default",
        report_ext="_classes",
        N_img=40,
        preprocessing_function=rtdc_toolbox.preprocess_rtdc_image,
        randomize=True,
        n_rows=4,
    )
