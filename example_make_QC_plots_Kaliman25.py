"""Example: create QC reports for Kaliman25-classified RTDC files.

For every RTDC file ending with ``extension_classified_rtdc`` this script
creates three types of quality-control output:

1. acquisition/class stability report;
2. example images from Kaliman classes;
3. class-colored scatter plots of final WBC populations.

All QC PNGs are stored below ``output_folder`` while preserving the relative
subfolder structure of ``main_folder_with_rtdc``.

The Kaliman classification itself is performed separately by
``example_Kaliman25_classification.py``.

rtdc-toolbox framework version 2.0 (August 2026)
"""

from __future__ import annotations

from pathlib import Path

import rtdc_toolbox


# =============================================================================
# USER SETTINGS
# =============================================================================

main_folder_with_rtdc = Path(r"Z:\KK\RTDC")
extension_classified_rtdc = "_kaliman25"
output_folder = Path(r"Z:\KK\QC_Kaliman25")
classifier_name = "Kaliman25_WBC"

# Replace existing QC PNGs when True.
overwrite = True

# Probability filtering used identically by class examples and scatter plots.
# Kaliman25 has no probability feature, so None disables the filter.
probability_threshold = None
keep_above_threshold = True

# -----------------------------------------------------------------------------
# Stability report settings
# -----------------------------------------------------------------------------

stability_features_to_plot = (
    "area_um",
    "deform",
    "bright_avg",
    "area_ratio",
)
stability_n_window = 10_000
stability_step = 1_000

# -----------------------------------------------------------------------------
# Class-example image settings
# -----------------------------------------------------------------------------

class_examples_N_img = 40
class_examples_randomize = True
class_examples_n_rows = 4

# -----------------------------------------------------------------------------
# Scatter-plot settings
# -----------------------------------------------------------------------------

classes_to_plot = [
    "lymphocytes",
    "neutrophils",
    "monocytes",
    "eosinophils",
]

plots_to_plot = [
    ["area_um", "deform"],
    ["area_um", "bright_avg"],
    ["area_um", "tex_den_avg"],
    ["area_um", "area_ratio"],
]

n_columns = 2
max_points_per_class = 20_000
point_size = 13.0
point_alpha = 0.75


# =============================================================================
# FIND FILES AND EXPORT QC REPORTS
# =============================================================================

rtdc_files = sorted(
    path
    for path in main_folder_with_rtdc.rglob(
        f"*{extension_classified_rtdc}.rtdc"
    )
    if path.is_file()
)

print(f"Found {len(rtdc_files)} RTDC files.\n")

for i, rtdc_path in enumerate(rtdc_files, start=1):
    print(f"[{i}/{len(rtdc_files)}] {rtdc_path}")

    relative_parent = rtdc_path.parent.relative_to(main_folder_with_rtdc)
    current_output_folder = output_folder / relative_parent
    current_output_folder.mkdir(parents=True, exist_ok=True)

    # All output paths are fully defined here, outside the toolbox functions.
    stability_png_path = current_output_folder / f"{rtdc_path.stem}_stability.png"
    class_examples_png_path = current_output_folder / f"{rtdc_path.stem}_classes.png"
    scatter_png_path = current_output_folder / f"{rtdc_path.stem}_scatter.png"

    # -------------------------------------------------------------------------
    # 1. Stability report
    # -------------------------------------------------------------------------

    if stability_png_path.exists() and not overwrite:
        print(f"  [SKIP stability] {stability_png_path}")
    else:
        try:
            result_png = rtdc_toolbox.export_rtdc_class_stability_report(
                rtdc_path=rtdc_path,
                output_png_path=stability_png_path,
                classifier_name=classifier_name,
                features_to_plot=stability_features_to_plot,
                n_window=stability_n_window,
                step=stability_step,
            )
            if result_png is not None:
                print(f"  Stability report: {result_png}")
        except Exception as exc:
            print(f"  [ERROR stability] {exc}")

    # -------------------------------------------------------------------------
    # 2. Example images from each Kaliman class
    # -------------------------------------------------------------------------

    if class_examples_png_path.exists() and not overwrite:
        print(f"  [SKIP class examples] {class_examples_png_path}")
    else:
        try:
            result_png = rtdc_toolbox.export_rtdc_class_examples(
                rtdc_path=rtdc_path,
                output_png_path=class_examples_png_path,
                classifier_name=classifier_name,
                N_img=class_examples_N_img,
                preprocessing_function=rtdc_toolbox.preprocess_rtdc_image,
                randomize=class_examples_randomize,
                probability_threshold=probability_threshold,
                keep_above_threshold=keep_above_threshold,
                n_rows=class_examples_n_rows,
            )
            if result_png is not None:
                print(f"  Class examples: {result_png}")
        except Exception as exc:
            print(f"  [ERROR class examples] {exc}")

    # -------------------------------------------------------------------------
    # 3. Class-colored scatter plots
    # -------------------------------------------------------------------------

    if scatter_png_path.exists() and not overwrite:
        print(f"  [SKIP scatter] {scatter_png_path}")
    else:
        try:
            result_png = rtdc_toolbox.export_rtdc_class_scatter_plots(
                rtdc_path=rtdc_path,
                output_png_path=scatter_png_path,
                classifier_name=classifier_name,
                classes_to_plot=classes_to_plot,
                plots_to_plot=plots_to_plot,
                n_columns=n_columns,
                probability_threshold=probability_threshold,
                keep_above_threshold=keep_above_threshold,
                max_points_per_class=max_points_per_class,
                point_size=point_size,
                point_alpha=point_alpha,
            )
            if result_png is not None:
                print(f"  Scatter plots: {result_png}")
        except Exception as exc:
            print(f"  [ERROR scatter] {exc}")

    print()

print("Done.")
