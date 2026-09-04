from pathlib import Path
import operator
import random

import dclab
import numpy as np
import rtdc_toolbox


# =============================================================================
# SETTINGS
# =============================================================================

INPUT_FOLDER = Path(r"Z:\KK_D979")
FILE_SUFFIX = "_dcn_b7c_kaliman25.rtdc"
ZIP_PATH = Path(r"Z:\KK\Training\filtered_random_images_D979.zip")

N_IMAGES_PER_FILE = 200

FILTERS = [
    ("userdef3", "==", 21),
    ("area_um", "<", 30),
]


# =============================================================================
# EXPORT
# =============================================================================

ops = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}

files = sorted(INPUT_FOLDER.rglob(f"*{FILE_SUFFIX}"))

if ZIP_PATH.exists():
    ZIP_PATH.unlink()

print(f"Found {len(files)} RTDC files.")

for path in files:

    with dclab.new_dataset(path) as ds:

        mask = np.ones(len(ds), dtype=bool)

        for feature, op, value in FILTERS:
            mask &= ops[op](np.asarray(ds[feature][:]), value)

        n_found = int(mask.sum())

    if n_found == 0:
        print(f"{path.name}: no matching events")
        continue

    n_export = min(N_IMAGES_PER_FILE, n_found)

    print(f"{path.name}: {n_found} matching, exporting {n_export}")

    rtdc_toolbox.export_preprocessed_images_from_rtdc_to_zip(
        rtdc_path=path,
        zip_path=ZIP_PATH,
        mask=mask,
        n=n_export,
        random_images=True,
        saving_mode="append",
    )

print(f"Done: {ZIP_PATH}")
