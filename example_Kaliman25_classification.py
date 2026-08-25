"""Example: Kaliman25 WBC classification only.

Set USE_IMAGE_SELECTED_WBCS to choose between two workflows:

False:
    Run the original Kaliman25 workflow on whole blood. Kaliman Step 1 performs
    the crude WBC preselection.

True:
    Read WBCs from the selected input RTDC file and pass that mask to Kaliman25
    with preselected_wbcs=True. Kaliman Step 1 is skipped and the image-selected
    WBCs enter directly at Step 2.

Quality-control plots are intentionally kept in the separate example script
``example_make_QC_plots_kaliman.py``.

rtdc-toolbox framework version 2.0 (August 2026)
"""

from __future__ import annotations

from pathlib import Path

import rtdc_toolbox


def find_unclassified_rtdc_files(
    folder: str | Path,
    input_extension: str,
    kaliman_extension: str,
) -> list[Path]:
    """Find input RTDC files that do not yet have a Kaliman-classified result."""
    folder = Path(folder)

    return sorted(
        path
        for path in folder.rglob(f"*{input_extension}.rtdc")
        if not path.with_name(
            f"{path.stem}{kaliman_extension}.rtdc"
        ).exists()
    )


# =============================================================================
# USER SETTINGS
# =============================================================================

main_folder_with_rtdc = r"Z:\KK"

# Analyze only RTDC files ending with this extension.
# Example input:
#     2026-04-29_14.57_M001_D003_dcn_b7c.rtdc
extension_rtdc_to_analyze = "_dcn_b7c"

# This is appended to the selected input filename.
# Example output:
#     2026-04-29_14.57_M001_D003_dcn_b7c_kaliman25.rtdc
extension_kaliman_rtdc = "_kaliman25"

USE_IMAGE_SELECTED_WBCS = True
WBC_PROBABILITY_THRESHOLD = 0.90

# Existing Kaliman RTDC files are replaced when True.
overwrite = True


# =============================================================================
# CLASSIFICATION
# =============================================================================

unclassified_rtdc_list = find_unclassified_rtdc_files(
    folder=main_folder_with_rtdc,
    input_extension=extension_rtdc_to_analyze,
    kaliman_extension=extension_kaliman_rtdc,
)

print(f"Found {len(unclassified_rtdc_list)} RTDC files to process.")

for i, path_to_rtdc in enumerate(unclassified_rtdc_list, start=1):
    print(
        f"Processing: {path_to_rtdc.relative_to(main_folder_with_rtdc)} "
        f"({i}/{len(unclassified_rtdc_list)})"
    )

    if USE_IMAGE_SELECTED_WBCS:
        # The selected input file is already the image-classified RTDC file.
        # Use its WBC class to build the mask for Kaliman25.
        wbc_mask = rtdc_toolbox.make_classifier_mask(
            path_or_ds=path_to_rtdc,
            classifier_name="default",
            classes="WBC",
            probability_threshold=WBC_PROBABILITY_THRESHOLD,
            probability_comparison=">",
        )

        print(f"WBC mask selected {int(wbc_mask.sum()):,} events.", flush=True)

        preselected_wbcs = True
        mask_description = (
            f"WBC events selected from {path_to_rtdc.name} with "
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
        overwrite=overwrite,
        verbose=True,
    )

    print("\nKaliman25-classifier metadata:\n", flush=True)
    rtdc_toolbox.print_rtdc_classifier_metadata(kaliman_rtdc)
