from pathlib import Path

import dclab
import numpy as np

import rtdc_toolbox


# =============================================================================
# USER SETTINGS
# =============================================================================

INPUT_FOLDER = Path(r"Z:\KK_Staro\Brez")
EXPORT_FOLDER = Path(r"D:\RTDC\ExportedClassesStaroBrez")

FILE_SUFFIX = "_dcn_b7c.rtdc"
CLASSIFIER_NAME = "default"

N_IMAGES_PER_CLASS = 100
PROBABILITY_THRESHOLD = 0.95


# =============================================================================
# EXPORT
# =============================================================================

EXPORT_FOLDER.mkdir(parents=True, exist_ok=True)

rtdc_files = sorted(INPUT_FOLDER.rglob(f"*{FILE_SUFFIX}"))

print(f"Found {len(rtdc_files)} RTDC files.\n")

for rtdc_path in rtdc_files:

    print(f"Processing: {rtdc_path}")

    classifier_info = rtdc_toolbox.get_classifier_metadata(
        rtdc_path,
        classifier_name=CLASSIFIER_NAME,
    )

    class_feature = classifier_info["class_feature"]
    probability_feature = classifier_info["probability_feature"]
    class_map = classifier_info["class_map"]

    with dclab.new_dataset(rtdc_path) as ds:
        classes = np.asarray(ds[class_feature][:], dtype=int)
        probabilities = np.asarray(ds[probability_feature][:], dtype=float)

    for class_id, class_name in class_map.items():

        mask = (
            (classes == int(class_id))
            & (probabilities > PROBABILITY_THRESHOLD)
        )

        n_available = int(mask.sum())

        if n_available == 0:
            continue

        print(
            f"  {class_name}: {n_available:,} events above "
            f"p>{PROBABILITY_THRESHOLD:g}"
        )

        rtdc_toolbox.export_preprocessed_images_from_rtdc_to_zip(
            rtdc_path=rtdc_path,
            zip_path=EXPORT_FOLDER / f"{class_name}.zip",
            mask=mask,
            n=N_IMAGES_PER_CLASS,
            random_images=True,
            saving_mode="append",
        )
