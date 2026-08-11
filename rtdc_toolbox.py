
"""Torch-free utilities for RT-DC (.rtdc) files.

rtdc-toolbox framework version 2.0 (August 2026)

Main public functions
---------------------
Image preparation and export
    get_img_diff
    preprocess_rtdc_image
    export_preprocessed_images_from_rtdc_to_zip

Classification and filtering
    classify_ds_WBC_Kaliman25
    classify_rtdc_WBC_Kaliman25
    classify_ds_images_ml          (lazy PyTorch wrapper)
    classify_rtdc_images_ml        (lazy PyTorch wrapper)
    make_classifier_mask

Image-model training and loading
    train_model_on_images          (lazy PyTorch wrapper)
    load_image_model               (lazy PyTorch wrapper)

Metadata and reporting
    get_classes_from_rtdc
    extract_rtdc_classifier_metadata
    get_classifier_metadata
    print_rtdc_classifier_metadata
    export_rtdc_class_stability_report
    export_rtdc_class_examples

Image-only PyTorch implementations live in ``rtdc_toolbox_ml.py``.  This core
module remains importable without torch/torchvision and exposes lazy wrappers
for the image-model functions.
"""

from __future__ import annotations

import os
import json
import copy
import zipfile
import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import cv2 as cv

from tqdm import tqdm
import random
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

import dclab

FRAMEWORK_VERSION = "2.0 (August 2026)"

__all__ = [
    "FRAMEWORK_VERSION",
    "get_img_diff",
    "preprocess_rtdc_image",
    "export_preprocessed_images_from_rtdc_to_zip",
    "save_features_to_rtdc",
    "classify_ds_WBC_Kaliman25",
    "classify_rtdc_WBC_Kaliman25",
    "classify_ds_images_ml",
    "classify_rtdc_images_ml",
    "train_model_on_images",
    "load_image_model",
    "get_classes_from_rtdc",
    "extract_rtdc_classifier_metadata",
    "get_classifier_metadata",
    "print_rtdc_classifier_metadata",
    "make_classifier_mask",
    "export_rtdc_class_stability_report",
    "export_rtdc_class_examples",
]


###########################################################################
#
# First we define the image preprocessing function (change it at will) 
#
###########################################################################

def get_img_diff(
    img1: np.ndarray,
    img2: np.ndarray,
    background_subtraction_shift: int = 128,
) -> np.ndarray:
    """Background subtraction with shift; output is uint8 with background ~128."""
    diff = img1.astype(np.int16) + background_subtraction_shift - img2.astype(np.int16)
    return np.clip(diff, 0, 255).astype(np.uint8)

def preprocess_rtdc_image(
    ds: Any,
    event_idx: int,
    extra_pixels: int = 15,
    no_overlap: bool = True,
) -> Optional[np.ndarray]:
    """
    Returns preprocessed image of event event_idx from dataset ds

    The preprocessing in this function comprises:
      - subtract background with get_img_diff (mean background ~128)
      - crop in x around the event contour
    input and output images are considered to be grayscale uint8
    Parameters
    ----------
    ds : dclab dataset-like
        Must provide "image", "image_bg", "frame", "contour".
    
    event_idx : int
        Index of the event to preprocess.
    extra_pixels : int
        Padding (in pixels) to add left/right of the contour.
    no_overlap : bool
        If False: just crop the cell to contour +/- extra_pixels in x direction.
        If True : reduce left/right padding when it would overlap with adjacent events in the same frame.
                 The crop borders are clamped to the x-midpoint between this event and its nearest
                 left/right neighbor (based on contour extents) within the same frame.
    """
    for k in ("image", "image_bg", "frame", "contour"):
        if k not in ds:
            print(f"ERROR: feature '{k}' not present in dataset")
            return None

    img = ds["image"][event_idx]
    img_bg = ds["image_bg"][event_idx]
    H, W = img.shape

    # background subtraction -> uint8 with background around 128
    img = get_img_diff(img, img_bg)

    # contour bounds (x-direction)
    x = ds["contour"][event_idx][:, 0]
    x0_min = int(np.floor(x.min()))
    x0_max = int(np.ceil(x.max()))

    # Padding (kept exactly for no_overlap=False)
    x_min = max(0, x0_min - extra_pixels)
    x_max = min(W, x0_max + extra_pixels)

    if no_overlap:
        # Events belonging to the same frame are stored contiguously in the RTDC file,
        # but their order within that frame is not guaranteed to be sorted by x.
        # We therefore:
        #   1) find the contiguous [lo, hi] block of events with the same frame id,
        #   2) compute contour x-extents for events in that block,
        #   3) pick the nearest neighbor on the left and on the right in *x* (not by index).

        frame_id = ds["frame"][event_idx]

        # Find contiguous block of events for this frame (scan outward from event_idx).
        lo = event_idx
        while lo > 0 and ds["frame"][lo - 1] == frame_id:
            lo -= 1
        hi = event_idx
        n_events = len(ds["frame"])
        while hi + 1 < n_events and ds["frame"][hi + 1] == frame_id:
            hi += 1

        # If there are other events in the same frame, find nearest left/right neighbors in x.
        if hi > lo:
            left_max = None   # maximum x (rightmost edge) among events strictly to the left
            right_min = None  # minimum x (leftmost edge) among events strictly to the right

            for j in range(lo, hi + 1):
                if j == event_idx:
                    continue
                xj = ds["contour"][j][:, 0]
                j_min = int(np.floor(xj.min()))
                j_max = int(np.ceil(xj.max()))

                # Candidate left neighbor: event whose right edge is at/before our left edge.
                if j_max <= x0_min:
                    if left_max is None or j_max > left_max:
                        left_max = j_max

                # Candidate right neighbor: event whose left edge is at/after our right edge.
                if j_min >= x0_max:
                    if right_min is None or j_min < right_min:
                        right_min = j_min

            # Clamp left border to midpoint with nearest left neighbor, if padding would overlap.
            if left_max is not None:
                mid_left = 0.5 * (left_max + x0_min)
                x_min = max(x_min, int(np.floor(mid_left)))

            # Clamp right border to midpoint with nearest right neighbor, if padding would overlap.
            if right_min is not None:
                mid_right = 0.5 * (x0_max + right_min)
                x_max = min(x_max, int(np.ceil(mid_right)))

    if x_max <= x_min:
        # fallback: return full image if contour/crop is degenerate
        return img

    img = img[:, x_min:x_max]
    return img


###########################################################################
#
# Next we define the main toolbox functions
#
###########################################################################



def get_classes_from_rtdc(path_or_ds: Any) -> Dict[str, Dict[int, str]]:
    """Return the class map of every classifier registered in an RTDC dataset.

    Returns
    -------
    dict
        ``{classifier_name: {class_value: class_name, ...}, ...}`` in metadata
        insertion order.  An empty dictionary is returned when no classifiers
        are registered.
    """
    info = extract_rtdc_classifier_metadata(path_or_ds)
    return {
        str(item["classifier_name"]): dict(item["class_map"])
        for item in info["classifiers"]
    }

def export_preprocessed_images_from_rtdc_to_zip(
    rtdc_path: Union[str, Path],
    zip_path: Union[str, Path] = "export.zip",
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    extra_pixels: int = 15,
    no_overlap: bool = True,
    img_ext: str = ".png",
    n: int = 1000,
    random_images: bool = False,
    saving_mode: str = "overwrite",
) -> None:
    """
    Export preprocessed event images from an RTDC file into a ZIP archive.

    Parameters
    ----------
    rtdc_path:
        Input RTDC file.
    zip_path:
        Output ZIP archive.
    mask:
        Optional Boolean mask with one value per event. If None, all events
        are eligible for export.
    extra_pixels:
        Extra pixels included to the left and right of the event contour.
    no_overlap:
        If True, preprocessing avoids overlap with neighbouring events.
    img_ext:
        Image format used inside the ZIP, e.g. ".png".
    n:
        Maximum number of images to export.
    random_images:
        If True, randomly select up to ``n`` events. Otherwise export the
        first ``n`` selected events.
    saving_mode:
        How to handle an existing ZIP archive:

        - "overwrite": replace the existing ZIP.
        - "append": keep the existing ZIP and add the new images.

        In append mode, a new ZIP is created if it does not yet exist.
    """
    saving_mode = str(saving_mode).strip().lower()

    if saving_mode == "overwrite":
        zip_mode = "w"
    elif saving_mode == "append":
        zip_mode = "a"
    else:
        raise ValueError(
            "saving_mode must be either 'overwrite' or 'append'"
        )

    with dclab.new_dataset(rtdc_path) as ds:

        if mask is None:
            idxs = np.arange(len(ds), dtype=int)
        else:
            mask = np.asarray(mask, dtype=bool)

            if mask.shape != (len(ds),):
                raise ValueError(
                    f"mask must have shape ({len(ds)},); "
                    f"got {mask.shape}"
                )

            idxs = np.where(mask)[0].astype(int)

            if len(idxs) == 0:
                print("No events selected (mask contains no True values).")
                return

        if random_images and len(idxs) > n:
            idxs = np.asarray(
                random.sample(list(idxs), k=int(n)),
                dtype=int,
            )
        else:
            idxs = idxs[:int(n)]

        ext = img_ext if img_ext.startswith(".") else "." + img_ext
        base_name = Path(rtdc_path).stem

        print(
            f"Exporting {len(idxs):,} images to ZIP "
            f"({saving_mode} mode): {zip_path}"
        )

        n_written = 0

        with zipfile.ZipFile(
            zip_path,
            zip_mode,
            zipfile.ZIP_STORED,
            allowZip64=True,
        ) as zipf:

            for i in tqdm(idxs):
                img = preprocess_rtdc_image(
                    ds,
                    int(i),
                    extra_pixels=extra_pixels,
                    no_overlap=no_overlap,
                )

                if img is None:
                    continue

                ok, buf = cv.imencode(ext, img)

                if not ok:
                    print(f"WARNING: cv.imencode failed for event {i}")
                    continue

                filename = f"{base_name}-{i}{ext}"
                zipf.writestr(filename, buf.tobytes())
                n_written += 1

        print(f"Done. Wrote {n_written:,} images to: {zip_path}")


def save_features_to_rtdc(
    basin_rtdc_path: str,
    output_rtdc_path: str,
    feature_dict: Dict[str, np.ndarray],
    metadata: dict,
    overwrite: bool = True,
) -> str:
    """
    Create a new RTDC file linked to `basin_rtdc_path`, storing the supplied
    features and metadata.

    Classification metadata format 2.0 explicitly records the class and
    probability feature for every classifier, so this function no longer tries
    to infer or rewrite a single class feature automatically.
    """
    out_path = str(output_rtdc_path)
    if os.path.exists(out_path) and not overwrite:
        print(f"[SKIP] file already exists (you can set overwrite to True): {out_path}", flush=True)
        return out_path

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    basin_locs, basin_format = _get_ultimate_basin_locs_and_format(str(basin_rtdc_path))

    with dclab.new_dataset(basin_rtdc_path) as ds0:
        base_meta = _strip_basin_metadata(ds0.config.as_dict(pop_filtering=True))
        n = len(ds0)
        inherited_features = set(ds0.features)

    incoming_meta = _strip_basin_metadata(metadata if isinstance(metadata, dict) else {})
    meta = _deep_merge_dicts(base_meta, incoming_meta)

    for k, v in dict(feature_dict).items():
        arr = np.asarray(v)
        if arr.shape[0] != n:
            raise ValueError(f"Feature '{k}' length {arr.shape[0]} != number of events {n}")

    # Detect inconsistent v2 metadata early.
    registry = _load_classification_registry_from_user(meta.get("user", {}))
    available_features = inherited_features | {str(k) for k in feature_dict}
    for entry in registry.get("entries", {}).values():
        if not isinstance(entry, dict):
            continue
        cf = entry.get("class_feature")
        pf = entry.get("probability_feature")
        if cf and str(cf) not in available_features:
            raise KeyError(f"Classification metadata refers to missing class feature '{cf}'")
        if pf not in (None, "") and str(pf) not in available_features:
            raise KeyError(f"Classification metadata refers to missing probability feature '{pf}'")

    with dclab.RTDCWriter(out_path, mode="reset") as hw:
        hw.store_metadata(meta)
        for k, v in dict(feature_dict).items():
            hw.store_feature(str(k), np.asarray(v))

    with dclab.RTDCWriter(out_path) as hw:
        try:
            hw.store_basin(
                basin_name="original",
                basin_type="file",
                basin_format=basin_format,
                basin_locs=basin_locs,
                mapping="same",
            )
        except TypeError:
            hw.store_basin(
                basin_name="original",
                basin_type="file",
                basin_format=basin_format,
                basin_locs=basin_locs,
            )

    return out_path


def _normalize_event_mask(
    mask: Optional[Union[np.ndarray, Sequence[bool]]],
    n_events: int,
) -> np.ndarray:
    """Return a Boolean event mask with exactly one value per event."""
    n_events = int(n_events)
    if mask is None:
        return np.ones(n_events, dtype=bool)

    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.shape != (n_events,):
        raise ValueError(
            f"mask must have shape ({n_events},); got {mask_array.shape}"
        )
    return mask_array


def _make_selection_info(
    mask: Optional[Union[np.ndarray, Sequence[bool]]],
    normalized_mask: np.ndarray,
    *,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the common mask-selection metadata used by all classifiers."""
    n_total = int(normalized_mask.size)
    n_selected = int(normalized_mask.sum())
    return {
        "mode": "all_events" if mask is None else "boolean_mask",
        "n_total_events": n_total,
        "n_selected_events": n_selected,
        "n_excluded_events": int(n_total - n_selected),
        "description": None if description in (None, "") else str(description),
    }


def _make_output_rtdc_path(
    path_to_rtdc: Union[str, Path],
    output_suffix: str,
) -> Path:
    """Return the output RTDC path produced by appending a suffix to the stem."""
    input_path = Path(path_to_rtdc)
    suffix = str(output_suffix)
    if not suffix:
        raise ValueError("output_suffix must not be empty")

    if input_path.suffix.lower() == ".rtdc":
        output_path = input_path.with_name(input_path.stem + suffix + ".rtdc")
    else:
        output_path = Path(str(input_path) + suffix + ".rtdc")

    if output_path.resolve() == input_path.resolve():
        raise ValueError("Output RTDC path must differ from the input path")
    return output_path




def classify_rtdc_WBC_Kaliman25(
    path_to_rtdc: str,
    *,
    classifier_name: Optional[str] = None,
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    mask_description: Optional[str] = None,
    preselected_wbcs: bool = False,
    output_suffix: str = "_kaliman25",
    overwrite: bool = False,
    verbose: bool = True,
) -> str:
    """Classify an RTDC file with the Kaliman 2025 WBC classifier.

    The dataset-level classifier performs the classification. This wrapper
    creates the output path, allocates the next free ``userdefN`` class feature,
    registers classifier metadata, and writes the basin-linked RTDC file.

    Set ``preselected_wbcs=True`` when ``mask`` already isolates WBCs, for
    example using an image classifier. In that mode Kaliman Step 1 (the crude
    area/deformation/area-ratio WBC prefilter) is skipped. Later Kaliman steps
    are unchanged.
    """
    input_path = Path(path_to_rtdc)
    output_path = _make_output_rtdc_path(input_path, output_suffix)

    if output_path.exists() and not overwrite:
        if verbose:
            print(f"[SKIP] File already exists: {output_path}", flush=True)
        return str(output_path)

    with dclab.new_dataset(str(input_path)) as ds:
        classes, _, result = classify_ds_WBC_Kaliman25(
            ds,
            mask=mask,
            classifier_name=classifier_name,
            mask_description=mask_description,
            preselected_wbcs=preselected_wbcs,
            verbose=verbose,
        )
        metadata = ds.config.as_dict(pop_filtering=True)

    class_feature = _allocate_userdef_features(str(input_path), 1)[0]
    if verbose:
        print(
            f"Classifier '{result['classifier_name']}' will store classes in "
            f"'{class_feature}'.",
            flush=True,
        )
        print(f"Writing classified RTDC file: {output_path}", flush=True)

    metadata = _register_classifier_metadata(
        metadata,
        class_feature=class_feature,
        probability_feature=None,
        class_map=result["class_map"],
        classifier=result["classifier"],
    )
    save_features_to_rtdc(
        basin_rtdc_path=str(input_path),
        output_rtdc_path=str(output_path),
        feature_dict={class_feature: classes},
        metadata=metadata,
        overwrite=True,
    )

    if verbose:
        print(f"Finished: {output_path}", flush=True)
    return str(output_path)





def classify_ds_WBC_Kaliman25(
    ds: Any,
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    *,
    classifier_name: Optional[str] = None,
    mask_description: Optional[str] = None,
    preselected_wbcs: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Classify selected events in an open dataset using Kaliman 2025.

    Events outside ``mask`` remain class 0. The function does not allocate RTDC
    features, register metadata, or write files.

    Parameters
    ----------
    preselected_wbcs:
        If False (default), apply the original Kaliman Step 1 crude WBC
        prefilter based on area, deformation, and area ratio. If True, ``mask``
        is assumed to already contain WBCs and Step 1 is skipped. A mask is
        required in this mode. All later Kaliman steps are unchanged.

    Returns
    -------
    classes, probabilities, result
        ``classes`` is a full-length ``uint8`` array. ``probabilities`` is
        ``None`` because this classifier does not produce confidence values.
        ``result`` contains the class map and classifier metadata.
    """
    # Classification thresholds and random seed.
    AREA_UM_MIN = 30.0
    AREA_UM_MAX = 100.0
    DEFORM_MAX = 0.12
    AREA_RATIO_MAX = 1.10
    BRIGHT_P90_MIN = -10.0
    BRIGHT_P90_MAX = 15.0
    DEFORM_RAW_MIN = 0.06
    DEFORM_RAW_MAX = 0.26
    TEX_CONTRAST_MAX = 250.0
    RANDOM_SEED = 42

    class_values = {
        "unclassified": 0,
        "lymphocytes": 21,
        "neutrophils": 22,
        "monocytes": 23,
        "eosinophils": 24,
        "step1_discarded": 91,
        "step2_discarded": 92,
        "step4_discarded": 94,
        "step6_discarded": 96,
    }
    class_map = {int(value): str(name) for name, value in class_values.items()}

    resolved_name = str(classifier_name or "WBC_Kaliman25").strip()
    if not resolved_name:
        raise ValueError("classifier_name must not be empty")

    if preselected_wbcs and mask is None:
        raise ValueError(
            "preselected_wbcs=True requires a mask containing preselected WBC events"
        )

    n_events = len(ds)
    requested_mask = _normalize_event_mask(mask, n_events)
    selection_info = _make_selection_info(
        mask,
        requested_mask,
        description=mask_description,
    )
    classes = np.zeros(n_events, dtype=np.uint8)

    if verbose:
        print(f"Total events: {n_events:,}", flush=True)
        if mask is not None:
            print(
                f"Events selected by mask: {selection_info['n_selected_events']:,}",
                flush=True,
            )
            print(
                f"Events excluded by mask: {selection_info['n_excluded_events']:,}",
                flush=True,
            )

    processing_info = {
        "n_total_events": int(n_events),
        "n_requested_events": int(requested_mask.sum()),
        "n_processed_events": int(requested_mask.sum()),
        "n_excluded_by_mask": int(n_events - requested_mask.sum()),
        "n_failed_events": 0,
    }
    classifier_info = {
        "name": resolved_name,
        "method": "Kaliman25",
        "model_type": "rules_and_clustering",
        "input_modality": "event_features",
        "publication": "Kaliman et al., Biophysical Journal, 2025",
        "metadata_version": CLASSIFICATION_METADATA_VERSION,
        "preselected_wbcs": bool(preselected_wbcs),
        "selection": selection_info,
        "processing": processing_info,
    }
    result = {
        "classifier_name": resolved_name,
        "class_map": class_map,
        "classifier": classifier_info,
    }

    if not np.any(requested_mask):
        return classes, None, result

    # Load all features used by the classifier.
    required_features = [
        "area_um",
        "area_um_raw",
        "area_ratio",
        "deform",
        "deform_raw",
        "bright_perc_90",
        "bright_avg",
        "tex_con_avg",
        "tex_den_avg",
        "tex_sva_avg",
        "tex_idm_ptp",
        "tex_den_ptp",
    ]
    missing_features = [
        feature for feature in required_features if feature not in ds.features
    ]
    if missing_features:
        raise KeyError(
            "Kaliman 2025 WBC classifier requires these missing features: "
            + ", ".join(missing_features)
        )

    area_um = np.asarray(ds["area_um"][:])
    area_um_raw = np.asarray(ds["area_um_raw"][:])
    area_ratio = np.asarray(ds["area_ratio"][:])
    deform = np.asarray(ds["deform"][:])
    deform_raw = np.asarray(ds["deform_raw"][:])
    bright_p90 = np.asarray(ds["bright_perc_90"][:])
    bright_avg = np.asarray(ds["bright_avg"][:])
    tex_con_avg = np.asarray(ds["tex_con_avg"][:])
    tex_den_avg = np.asarray(ds["tex_den_avg"][:])
    tex_sva_avg = np.asarray(ds["tex_sva_avg"][:])
    tex_idm_ptp = np.asarray(ds["tex_idm_ptp"][:])
    tex_den_ptp = np.asarray(ds["tex_den_ptp"][:])

    labels = np.zeros(n_events, dtype=np.int32)

    # Step 1: Select crude WBC candidates, unless WBCs were preselected upstream.
    if preselected_wbcs:
        crude_wbcs = requested_mask.copy()
        step1_discarded = np.zeros(n_events, dtype=bool)
        if verbose:
            print(
                f"Step 1 skipped: using {int(crude_wbcs.sum()):,} "
                "preselected WBC events from mask.",
                flush=True,
            )
    else:
        crude_wbcs = requested_mask & (
            (area_um >= AREA_UM_MIN)
            & (area_um <= AREA_UM_MAX)
            & (deform <= DEFORM_MAX)
            & (area_ratio <= AREA_RATIO_MAX)
        )
        step1_discarded = requested_mask & ~crude_wbcs
        labels[step1_discarded] = class_values["step1_discarded"]
        if verbose:
            print(
                f"Step 1 kept: {int(crude_wbcs.sum()):,} "
                f"(discarded: {int(step1_discarded.sum()):,})",
                flush=True,
            )

    # Step 2a: Remove RBC-like events with a two-component GMM.
    step2_base = crude_wbcs
    step2_indices = np.flatnonzero(step2_base)
    if step2_indices.size >= 2:
        step2_data = np.column_stack(
            (tex_idm_ptp[step2_base], tex_den_ptp[step2_base])
        )
        step2_model = GaussianMixture(
            n_components=2,
            random_state=RANDOM_SEED,
        ).fit(step2_data)
        step2_prediction = step2_model.predict(step2_data)
        rbc_cluster = int(np.argmax(np.sum(step2_model.means_ ** 2, axis=1)))
        wbc_cluster = 1 - rbc_cluster
        step2a_wbcs = np.zeros(n_events, dtype=bool)
        step2a_wbcs[step2_indices] = step2_prediction == wbc_cluster
    elif step2_indices.size == 1:
        step2a_wbcs = step2_base.copy()
    else:
        step2a_wbcs = np.zeros(n_events, dtype=bool)
    if verbose:
        print(f"Step 2a kept: {int(step2a_wbcs.sum()):,}", flush=True)

    # Step 2b: Apply the brightness-percentile window.
    step2b_wbcs = (
        step2a_wbcs
        & (bright_p90 >= BRIGHT_P90_MIN)
        & (bright_p90 <= BRIGHT_P90_MAX)
    )
    if verbose:
        print(f"Step 2b kept: {int(step2b_wbcs.sum()):,}", flush=True)

    # Step 2c: Apply the raw-deformation window.
    wbcs = (
        step2b_wbcs
        & (deform_raw >= DEFORM_RAW_MIN)
        & (deform_raw <= DEFORM_RAW_MAX)
    )
    step2_discarded = step2_base & ~wbcs
    labels[step2_discarded] = class_values["step2_discarded"]
    if verbose:
        print(
            f"Step 2 kept: {int(wbcs.sum()):,} "
            f"(discarded: {int(step2_discarded.sum()):,})",
            flush=True,
        )

    # Step 3: Split crude lymphocytes from granulocyte/monocyte candidates.
    step3_indices = np.flatnonzero(wbcs)
    crude_lymphocytes = np.zeros(n_events, dtype=bool)
    if step3_indices.size >= 2:
        step3_data = np.column_stack((area_um[wbcs], area_um_raw[wbcs]))
        step3_model = KMeans(
            n_clusters=2,
            random_state=RANDOM_SEED,
            n_init="auto",
        )
        step3_prediction = step3_model.fit_predict(step3_data)
        lymphocyte_cluster = int(np.argmin(step3_model.cluster_centers_.mean(axis=1)))
        crude_lymphocytes[step3_indices] = step3_prediction == lymphocyte_cluster
    elif step3_indices.size == 1:
        crude_lymphocytes[step3_indices[0]] = True
    granulocyte_monocyte_candidates = wbcs & ~crude_lymphocytes
    if verbose:
        print(
            f"Step 3 crude lymphocytes: {int(crude_lymphocytes.sum()):,}",
            flush=True,
        )
        print(
            "Step 3 granulocyte/monocyte candidates: "
            f"{int(granulocyte_monocyte_candidates.sum()):,}",
            flush=True,
        )

    # Step 4: Clean the lymphocyte branch.
    step4_candidates = crude_lymphocytes & (tex_con_avg <= TEX_CONTRAST_MAX)
    step4_indices = np.flatnonzero(step4_candidates)
    lymphocytes = np.zeros(n_events, dtype=bool)
    if step4_indices.size >= 2:
        step4_data = np.column_stack(
            (bright_avg[step4_candidates], tex_con_avg[step4_candidates])
        )
        step4_model = GaussianMixture(
            n_components=2,
            random_state=RANDOM_SEED,
        ).fit(step4_data)
        step4_prediction = step4_model.predict(step4_data)
        lymphocyte_cluster = int(np.argmax(step4_model.means_[:, 0]))
        lymphocytes[step4_indices] = step4_prediction == lymphocyte_cluster
    elif step4_indices.size == 1:
        lymphocytes[step4_indices[0]] = True
    labels[lymphocytes] = class_values["lymphocytes"]
    step4_discarded = crude_lymphocytes & ~lymphocytes
    labels[step4_discarded] = class_values["step4_discarded"]
    if verbose:
        print(
            f"Step 4 lymphocytes: {int(lymphocytes.sum()):,} "
            f"(discarded: {int(step4_discarded.sum()):,})",
            flush=True,
        )

    # Step 5: Identify neutrophils as the largest of three clusters.
    step5_indices = np.flatnonzero(granulocyte_monocyte_candidates)
    neutrophils = np.zeros(n_events, dtype=bool)
    if step5_indices.size >= 3:
        step5_data = np.column_stack(
            (
                bright_avg[granulocyte_monocyte_candidates],
                tex_den_avg[granulocyte_monocyte_candidates],
                tex_sva_avg[granulocyte_monocyte_candidates],
            )
        )
        step5_model = GaussianMixture(
            n_components=3,
            random_state=RANDOM_SEED,
        ).fit(step5_data)
        step5_prediction = step5_model.predict(step5_data)
        cluster_counts = np.bincount(step5_prediction, minlength=3)
        neutrophil_cluster = int(np.argmax(cluster_counts))
        neutrophils[step5_indices] = step5_prediction == neutrophil_cluster
    elif step5_indices.size > 0:
        neutrophils[step5_indices] = True
    labels[neutrophils] = class_values["neutrophils"]
    monocyte_eosinophil_candidates = granulocyte_monocyte_candidates & ~neutrophils
    if verbose:
        print(f"Step 5 neutrophils: {int(neutrophils.sum()):,}", flush=True)
        print(
            "Step 5 monocyte/eosinophil candidates: "
            f"{int(monocyte_eosinophil_candidates.sum()):,}",
            flush=True,
        )

    # Step 6: Separate monocytes and eosinophils by neutrophil texture range.
    monocytes = np.zeros(n_events, dtype=bool)
    eosinophils = np.zeros(n_events, dtype=bool)
    if neutrophils.any():
        neutrophil_texture = tex_den_avg[neutrophils]
        eosinophils = monocyte_eosinophil_candidates & (
            tex_den_avg > float(np.max(neutrophil_texture))
        )
        monocytes = monocyte_eosinophil_candidates & (
            tex_den_avg < float(np.min(neutrophil_texture))
        )
    labels[monocytes] = class_values["monocytes"]
    labels[eosinophils] = class_values["eosinophils"]
    step6_discarded = (
        monocyte_eosinophil_candidates & ~monocytes & ~eosinophils
    )
    labels[step6_discarded] = class_values["step6_discarded"]
    if verbose:
        print(f"Step 6 monocytes: {int(monocytes.sum()):,}", flush=True)
        print(f"Step 6 eosinophils: {int(eosinophils.sum()):,}", flush=True)
        print(f"Step 6 discarded: {int(step6_discarded.sum()):,}", flush=True)

    classes[requested_mask] = labels[requested_mask].astype(np.uint8, copy=False)
    return classes, None, result




def export_rtdc_class_stability_report(
    path_to_rtdc: Union[str, Path],
    classifier_name: Optional[str] = "default",
    features_to_plot: Sequence[str] = ("area_um", "deform"),
    n_window: int = 10000,
    step: int = 1000,
    out_png: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """
    Analyze one RTDC file and save a PNG stability report.

    The first row gives classifier-independent acquisition QC:
      - number of events per ``n_window`` frames,
      - temperature (``temp``),
      - background-corrected brightness (``bright_bc_avg``),
      - lateral position (``pos_y``).

    Remaining rows show, for each class:
      - percentage of events in the class,
      - mean ± SD of each feature in ``features_to_plot``.

    ``classifier_name="default"`` selects the first metadata entry. Pass a
    concrete classifier name to select it explicitly, or ``None`` to analyze
    all events as one group.
    """
    rtdc_path = Path(path_to_rtdc)
    if not rtdc_path.exists():
        print(f"[SKIP] RTDC not found: {rtdc_path}")
        return None

    features_to_plot = list(features_to_plot)
    out_png = Path(out_png) if out_png is not None else rtdc_path.with_suffix(".png")

    def get_time(ds):
        return (
            np.asarray(ds["time"], float)
            if "time" in ds.features
            else np.arange(len(ds), dtype=float)
        )

    def slices(n, w, st):
        st = w if st is None else st
        out = []
        i = 0
        while i < n:
            j = min(i + w, n)
            out.append(slice(i, j))
            if j == n:
                break
            i += st
        return out

    with dclab.new_dataset(rtdc_path) as ds:
        t = get_time(ds)
        n_total_events = len(ds)

        if classifier_name is None:
            cls = np.zeros(len(ds), dtype=int)
            local_names = {0: "All events"}
            class_feature = None
        else:
            cinfo = get_classifier_metadata(ds, classifier_name)
            resolved_classifier_name = cinfo["classifier_name"]
            class_feature = cinfo["class_feature"]
            if class_feature not in ds.features:
                print(
                    f"[SKIP] {rtdc_path.name}: "
                    f"class feature '{class_feature}' not found"
                )
                return None
            cls = np.asarray(ds[class_feature], dtype=int)
            local_names = cinfo["class_map"]

        if len(cls) == 0:
            print(f"[SKIP] {rtdc_path.name}: no events")
            return None

        # Event-index windows used for the all-event feature QC and class-wise plots.
        win = slices(len(cls), n_window, step)
        t_centers = np.array([np.nanmean(t[s]) for s in win], float)

        # Requested class-wise feature arrays. Missing features become NaN.
        feat_arr = {
            f: (
                np.asarray(ds[f], float)
                if f in ds.features
                else np.full(len(cls), np.nan, float)
            )
            for f in features_to_plot
        }

        # First-row acquisition QC features, independent of classifier.
        overview_features = ["temp", "bright_bc_avg", "pos_y"]
        overview_arr = {
            f: (
                np.asarray(ds[f], float)
                if f in ds.features
                else np.full(len(cls), np.nan, float)
            )
            for f in overview_features
        }

        # Event-count stability is evaluated in frame windows.
        if "frame" in ds.features:
            frame = np.asarray(ds["frame"], int)
            frame_min = int(np.nanmin(frame))
            frame_max = int(np.nanmax(frame))

            frame_starts = np.arange(frame_min, frame_max + 1, int(step))
            frame_stops = frame_starts + int(n_window)

            n_events_per_frame_window = np.array(
                [
                    np.sum((frame >= f0) & (frame < f1))
                    for f0, f1 in zip(frame_starts, frame_stops)
                ],
                float,
            )

            t_frame_centers = np.array(
                [
                    np.nanmean(t[(frame >= f0) & (frame < f1)])
                    if np.any((frame >= f0) & (frame < f1))
                    else np.nan
                    for f0, f1 in zip(frame_starts, frame_stops)
                ],
                float,
            )
        else:
            # Fallback for unusual RTDC files without a frame feature.
            frame = None
            n_events_per_frame_window = np.array(
                [s.stop - s.start for s in win],
                float,
            )
            t_frame_centers = t_centers

        classes = np.unique(cls)

        n_overview_cols = 1 + len(overview_features)  # event count + 3 QC features
        n_class_cols = 1 + len(features_to_plot)
        n_cols = max(n_overview_cols, n_class_cols)
        n_rows = 1 + len(classes)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.8 * n_cols, max(3.0, 1.9 * n_rows)),
            sharex=True,
        )

        if n_rows == 1:
            axes = np.array([axes])
        if n_cols == 1:
            axes = axes.reshape(n_rows, 1)

        # =====================================================================
        # First row: classifier-independent acquisition QC
        # =====================================================================

        # Events per frame window.
        ax = axes[0, 0]
        ax.plot(t_frame_centers, n_events_per_frame_window)
        if frame is not None:
            ax.set_title(f"Events per {n_window} frames")
        else:
            ax.set_title("Events per window")
        ax.set_ylabel("N events")
        ax.grid(True, alpha=0.25)

        # temp, bright_bc_avg, pos_y: all-event mean ± SD over time.
        for ci, f in enumerate(overview_features, start=1):
            ax = axes[0, ci]
            vals = overview_arr[f]

            means = np.full(len(win), np.nan, float)
            sds = np.full(len(win), np.nan, float)

            for wi, s in enumerate(win):
                v = vals[s]
                if v.size:
                    means[wi] = np.nanmean(v)
                    sds[wi] = np.nanstd(v)

            overall = np.nanmean(vals)
            if np.isfinite(overall):
                ax.axhline(overall, color="red", lw=1, alpha=0.8)
                ax.text(
                    0.98,
                    0.98,
                    f"avg={overall:.3g}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    color="red",
                    fontsize=8,
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.7,
                    ),
                )

            ax.plot(t_centers, means)
            if np.isfinite(sds).any():
                ax.fill_between(
                    t_centers,
                    means - sds,
                    means + sds,
                    alpha=0.2,
                )

            ax.set_title(f"All events: {f}")
            ax.set_ylabel(f)
            ax.grid(True, alpha=0.25)

        # Hide any unused cells in the first row.
        for ci in range(n_overview_cols, n_cols):
            axes[0, ci].axis("off")

        # =====================================================================
        # Remaining rows: class-wise stability
        # =====================================================================

        for ri, c in enumerate(classes, start=1):
            cname = local_names.get(int(c), str(int(c)))

            ax = axes[ri, 0]
            y = np.array(
                [100.0 * np.mean(cls[s] == c) for s in win],
                float,
            )
            overall = 100.0 * np.mean(cls == c)

            ax.axhline(overall, color="red", lw=1, alpha=0.8)
            ax.text(
                0.98,
                0.98,
                f"avg={overall:.2f}%",
                transform=ax.transAxes,
                ha="right",
                va="top",
                color="red",
                fontsize=8,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.7,
                ),
            )
            ax.plot(t_centers, y)

            if ri == 1:
                ax.set_title("% of events in window")

            ax.set_ylabel(f"{cname}\nPercent (%)")
            ax.grid(True, alpha=0.25)

            if ri == n_rows - 1:
                ax.set_xlabel("Time")

            for ci, f in enumerate(features_to_plot, start=1):
                ax = axes[ri, ci]
                means = np.full(len(win), np.nan, float)
                sds = np.full(len(win), np.nan, float)
                vals = feat_arr[f]

                for wi, s in enumerate(win):
                    v = vals[s][cls[s] == c]
                    if v.size:
                        means[wi] = np.nanmean(v)
                        sds[wi] = np.nanstd(v)

                overall = (
                    np.nanmean(vals[cls == c])
                    if np.any(cls == c)
                    else np.nan
                )

                if np.isfinite(overall):
                    ax.axhline(overall, color="red", lw=1, alpha=0.8)
                    ax.text(
                        0.98,
                        0.98,
                        f"avg={overall:.3g}",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        color="red",
                        fontsize=8,
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            facecolor="white",
                            edgecolor="none",
                            alpha=0.7,
                        ),
                    )

                ax.plot(t_centers, means)

                if np.isfinite(sds).any():
                    ax.fill_between(
                        t_centers,
                        means - sds,
                        means + sds,
                        alpha=0.2,
                    )

                if ri == 1:
                    ax.set_title(f"Mean ± SD: {f}")

                ax.set_ylabel(f)
                ax.grid(True, alpha=0.25)

                if ri == n_rows - 1:
                    ax.set_xlabel("Time")

            # Hide unused cells if the overview row requires more columns than
            # this class-wise row.
            for ci in range(n_class_cols, n_cols):
                axes[ri, ci].axis("off")

        label = (
            "all events"
            if classifier_name is None
            else (
                f"classifier={resolved_classifier_name}, "
                f"class_feature={class_feature}"
            )
        )

        fig.suptitle(
            f"{rtdc_path.name} | N total = {n_total_events} | "
            f"{label} | window={n_window}",
            y=0.995,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.98])

        try:
            fig.savefig(out_png, dpi=200)
            plt.close(fig)
            return out_png
        except Exception as e:
            plt.close(fig)
            print(f"[FAIL] Could not save {out_png}: {e}")
            return None


def export_rtdc_class_examples(
    path_to_rtdc: Union[str, Path],
    classifier_name: str = "default",
    report_ext: str = "_classes",
    N_img: int = 30,
    preprocessing_function=None,
    randomize: bool = True,
    prob_thresh: float = 0.9,
    keep_above_thresh: bool = True,
    n_rows: int = 2,
    # --- new fine-tuning parameters ---
    title_font_size: int = 16,
    class_header_font_size: int = 14,
    event_font_size: int = 10,
    vgap_rows_px: int = 6,            # vertical distance between image rows (within a class)
    hgap_images_px: int = 6,          # horizontal distance between images
    title_to_first_px: int = 12,      # vertical distance from main title to first class header
    # event number overlay placement (upper-right corner of the image area)
    event_inset_x_px: int = 4,        # distance from right edge of image (pixels)
    event_inset_y_px: int = 4,        # distance from top edge of image (pixels)
    event_text_bg: bool = False,       # draw a small background box behind event text
    event_text_bg_pad_px: int = 2,    # padding around event text background
) -> Optional[Path]:
    """
    Create a PNG report showing example preprocessed cell images for each class.

    Probability filtering is applied to classified classes only. Class 0 is the
    metadata-defined unclassified class and normally has probability 0, so its
    examples are always sampled without applying ``prob_thresh``.

    Uses PIL for pixel-exact spacing and overlay text.

    Fine-tuning knobs:
      1) title_font_size: font size for the main title
      2) event_font_size: font size for event numbers
      3) vgap_rows_px: vertical gap between image rows
      4) hgap_images_px: horizontal gap between images
      5) title_to_first_px: vertical gap from main title to first class header
      6) event numbers are drawn ON the image (upper-right), with configurable insets
    """
    from PIL import Image, ImageDraw, ImageFont
    import textwrap

    rtdc_path = Path(path_to_rtdc)
    if not rtdc_path.exists():
        print(f"[SKIP] RTDC not found: {rtdc_path}")
        return None

    if preprocessing_function is None:
        preprocessing_function = preprocess_rtdc_image

    # Basic validation / normalization
    N_img = int(N_img)
    n_rows = int(n_rows)
    if N_img <= 0:
        print("[SKIP] N_img must be > 0")
        return None
    if n_rows <= 0:
        print("[SKIP] n_rows must be > 0")
        return None

    title_font_size = int(title_font_size)
    class_header_font_size = int(class_header_font_size)
    event_font_size = int(event_font_size)
    vgap_rows_px = max(0, int(vgap_rows_px))
    hgap_images_px = max(0, int(hgap_images_px))
    title_to_first_px = max(0, int(title_to_first_px))
    event_inset_x_px = max(0, int(event_inset_x_px))
    event_inset_y_px = max(0, int(event_inset_y_px))
    event_text_bg_pad_px = max(0, int(event_text_bg_pad_px))

    # Output name: "<stem><report_ext>.png"
    stem = rtdc_path.name
    if stem.lower().endswith(".rtdc"):
        stem = stem[:-5]
    out_png = rtdc_path.with_name(f"{stem}{report_ext}.png")

    # Resolve features and class names from metadata using classifier_name.
    classifier_info = get_classifier_metadata(str(rtdc_path), classifier_name)
    class_feature = classifier_info["class_feature"]
    prob_feature = classifier_info["probability_feature"]
    class_names_meta = classifier_info["class_map"]

    # --- Fonts ---
    # Best effort: try a common TrueType font; fallback to PIL default if not available.
    def _load_font(size: int) -> ImageFont.ImageFont:
        for name in ("DejaVuSans.ttf", "arial.ttf", "LiberationSans-Regular.ttf"):
            try:
                return ImageFont.truetype(name, size=size)
            except Exception:
                pass
        return ImageFont.load_default()

    font_title = _load_font(title_font_size)
    font_class = _load_font(class_header_font_size)
    font_event = _load_font(event_font_size)

    # Helpers
    dummy = Image.new("L", (10, 10), 255)
    ddraw = ImageDraw.Draw(dummy)

    def _text_wh(draw: ImageDraw.ImageDraw, s: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
        if hasattr(draw, "textbbox"):
            bb = draw.textbbox((0, 0), s, font=font)
            return (bb[2] - bb[0], bb[3] - bb[1])
        return draw.textsize(s, font=font)

    def _pct(n: int, d: int) -> str:
        if d <= 0:
            return "0.0%"
        return f"{(100.0 * float(n) / float(d)):.1f}%"

    def _wrap_to_width(draw: ImageDraw.ImageDraw, s: str, max_px: int, font: ImageFont.ImageFont) -> List[str]:
        s = str(s).strip()
        if not s:
            return [""]
        if _text_wh(draw, s, font)[0] <= max_px:
            return [s]

        avg_char_w = max(4, _text_wh(draw, "ABCDEFGHIJKLMNOPQRSTUVWXYZ", font)[0] // 26)
        max_chars = max(10, max_px // avg_char_w)
        wrapped = textwrap.wrap(s, width=max_chars)

        out = []
        for line in wrapped:
            while _text_wh(draw, line, font)[0] > max_px and len(line) > 5:
                line = line[:-1]
            out.append(line)
        return out if out else [s[:max(1, max_chars)]]

    def _pad_to_hw_uint8(im: np.ndarray, H: int, W: int, fill: int = 128) -> np.ndarray:
        im = np.asarray(im)
        if im.ndim != 2:
            raise ValueError("Expected 2D grayscale image")
        if im.dtype != np.uint8:
            im = im.astype(np.uint8, copy=False)

        h, w = im.shape
        pad_h = max(0, H - h)
        pad_w = max(0, W - w)
        if pad_h > 0 or pad_w > 0:
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            im = cv.copyMakeBorder(
                im, top, bottom, left, right,
                borderType=cv.BORDER_CONSTANT,
                value=int(fill),
            )

        h2, w2 = im.shape
        if h2 > H:
            y0 = (h2 - H) // 2
            im = im[y0:y0 + H, :]
        if w2 > W:
            x0 = (w2 - W) // 2
            im = im[:, x0:x0 + W]
        return im

    with dclab.new_dataset(rtdc_path) as ds:
        n_total = len(ds)
        if n_total == 0:
            print(f"[SKIP] {rtdc_path.name}: no events")
            return None

        if class_feature not in ds.features:
            print(f"[SKIP] {rtdc_path.name}: class_feature '{class_feature}' not found")
            return None

        cls = np.asarray(ds[class_feature], dtype=int)
        classes = np.unique(cls)

        if prob_feature is None:
            # Classifier has no probability output: include every image in each class.
            have_prob = False
            above = np.zeros(n_total, dtype=bool)
            below = np.zeros(n_total, dtype=bool)
            sel_mask = np.ones(n_total, dtype=bool)
        else:
            if str(prob_feature) not in ds.features:
                print(f"[SKIP] {rtdc_path.name}: probability feature '{prob_feature}' not found")
                return None
            have_prob = True
            prob = np.asarray(ds[prob_feature], dtype=float)
            finite = np.isfinite(prob)
            above = finite & (prob >= float(prob_thresh))
            below = finite & (prob < float(prob_thresh))
            sel_mask = above if keep_above_thresh else below

        # Select indices per class
        per_class = []
        max_cols = 1
        all_selected = []

        for c in classes:
            c = int(c)
            in_class = (cls == c)
            n_in_class = int(in_class.sum())

            # Class 0 is reserved for unclassified events. Its probability is
            # normally 0 by design, so applying a high-confidence threshold
            # would make its report section blank even when many events exist.
            probability_filter_applied = bool(have_prob and c != 0)

            if have_prob:
                n_above = int((in_class & above).sum())
                n_below = int((in_class & below).sum())
            else:
                n_above = n_in_class
                n_below = 0

            class_selection_mask = sel_mask if probability_filter_applied else np.ones(n_total, dtype=bool)
            candidates = np.flatnonzero(in_class & class_selection_mask).astype(int)
            n_candidates = int(candidates.size)

            if n_candidates == 0:
                chosen = np.array([], dtype=int)
            else:
                if randomize and n_candidates > N_img:
                    chosen = np.asarray(random.sample(list(candidates), k=int(N_img)), dtype=int)
                else:
                    chosen = candidates[: int(N_img)]

            n_show = int(chosen.size)
            n_cols = int(np.ceil(max(1, n_show) / float(n_rows)))
            n_cols = max(1, n_cols)
            max_cols = max(max_cols, n_cols)

            cname = class_names_meta.get(c, str(c))
            per_class.append(
                dict(
                    c=c,
                    cname=str(cname),
                    n_in_class=n_in_class,
                    n_above=n_above,
                    n_below=n_below,
                    probability_filter_applied=probability_filter_applied,
                    chosen=chosen,
                    n_cols=n_cols,
                )
            )
            all_selected.extend(chosen.tolist())

        if len(all_selected) == 0:
            # Tiny text-only report
            W = 1200
            H = 220
            canvas = Image.new("L", (W, H), 255)
            draw = ImageDraw.Draw(canvas)
            lines = [
                f"{'Random examples' if randomize else 'Examples'} of images in classes | {rtdc_path.name}",
                f"N={n_total}",
                f"classifier='{classifier_name}' | class feature='{class_feature}'"
                ]
            if have_prob:
                side = ">=" if keep_above_thresh else "<"
                lines += [
                    f"probability feature='{prob_feature}'",
                    f"probability threshold: {side} {prob_thresh:g}",
                ]
            else:
                lines.append("probability feature=None (all images included)")
            y = 8
            for ln in lines:
                draw.text((8, y), ln, font=font_title, fill=0)
                y += _text_wh(draw, ln, font_title)[1] + 6
            canvas.save(out_png)
            return out_png

        # Render images & find max H/W
        imgs_by_idx: Dict[int, np.ndarray] = {}
        max_h = 1
        max_w = 1
        for i in all_selected:
            try:
                im = preprocessing_function(ds, int(i))
            except TypeError:
                im = preprocessing_function(ds, int(i), extra_pixels=15, no_overlap=True)

            if im is None:
                continue
            im = np.asarray(im)
            if im.ndim != 2:
                continue
            if im.dtype != np.uint8:
                im = im.astype(np.uint8, copy=False)

            imgs_by_idx[int(i)] = im
            h, w = im.shape
            max_h = max(max_h, int(h))
            max_w = max(max_w, int(w))

        if len(imgs_by_idx) == 0:
            print(f"[SKIP] {rtdc_path.name}: could not render any images")
            return None

        # Tile = exactly the image area (no extra index bar) because index is overlaid on the image
        tile_w = max_w
        tile_h = max_h

        # Margins: keep a small outer padding (reuse hgap as symmetric padding)
        outer_pad_x = hgap_images_px
        outer_pad_y = vgap_rows_px

        # Canvas width
        canvas_w = (max_cols * tile_w) + ((max_cols - 1) * hgap_images_px) + (2 * outer_pad_x)

        # Global title text
        if have_prob:
            side = ">=" if keep_above_thresh else "<"
            title = (
                f"{'Random examples' if randomize else 'Examples'} of images in classes | {rtdc_path.name}"
                f" | N={n_total} | classifier='{classifier_name}' | class feature='{class_feature}' | "
                f"probability feature='{prob_feature}' | probability threshold for classified classes: "
                f"{side} {prob_thresh:g} | class 0 included without probability filtering"
            )
        else:
            title = (
                f"{rtdc_path.name} | N={n_total} | classifier='{classifier_name}' | "
                f"class feature='{class_feature}' | probability feature=None (all images included)"
            )

        title_lines = _wrap_to_width(ddraw, title, max_px=canvas_w - 2 * outer_pad_x, font=font_title)
        title_line_h = _text_wh(ddraw, "Ag", font_title)[1]
        title_block_h = len(title_lines) * title_line_h

        # Per-class headers (wrapped)
        class_line_h = _text_wh(ddraw, "Ag", font_class)[1]

        class_blocks = []
        total_h = 0
        total_h += outer_pad_y
        total_h += title_block_h
        total_h += title_to_first_px  # custom: title -> first class header gap

        for d in per_class:
            nclass_pct_total = _pct(d["n_in_class"], n_total)

            if have_prob and not d["probability_filter_applied"]:
                header = (
                    f"{d['cname']} (class={d['c']}) | "
                    f"N_class={d['n_in_class']} ({nclass_pct_total}) | "
                    "probability filter not applied to unclassified events"
                )
            elif have_prob:
                header = (
                    f"{d['cname']} (class={d['c']}) | "
                    f"N_class={d['n_in_class']} ({nclass_pct_total}) | "
                    f"above={d['n_above']} ({_pct(d['n_above'], d['n_in_class'])}) | "
                    f"below={d['n_below']} ({_pct(d['n_below'], d['n_in_class'])})"
                )
            else:
                header = (
                    f"{d['cname']} (class={d['c']}) | "
                    f"N_class={d['n_in_class']} ({nclass_pct_total}) | prob n/a"
                )

            header_lines = _wrap_to_width(ddraw, header, max_px=canvas_w - 2 * outer_pad_x, font=font_class)
            header_h = len(header_lines) * class_line_h

            # Grid height for this class
            grid_h = (n_rows * tile_h) + ((n_rows - 1) * vgap_rows_px)

            # Store block; between classes we add 2*vgap_rows_px to keep it easy to tune
            class_blocks.append((d, header_lines, header_h, grid_h))
            total_h += header_h
            total_h += vgap_rows_px  # header -> first image row
            total_h += grid_h
            total_h += 2 * vgap_rows_px  # class gap

        total_h += outer_pad_y

        canvas = Image.new("L", (canvas_w, total_h), 255)
        draw = ImageDraw.Draw(canvas)

        y = outer_pad_y

        # Draw title
        for ln in title_lines:
            draw.text((outer_pad_x, y), ln, font=font_title, fill=0)
            y += title_line_h

        y += title_to_first_px

        # Draw classes
        for d, header_lines, header_h, grid_h in class_blocks:
            for ln in header_lines:
                draw.text((outer_pad_x, y), ln, font=font_class, fill=0)
                y += class_line_h

            y += vgap_rows_px  # header -> images

            chosen = [int(i) for i in d["chosen"] if int(i) in imgs_by_idx]
            pos = 0

            for rr in range(n_rows):
                row_y = y + rr * (tile_h + (vgap_rows_px if rr > 0 else 0))
                if rr > 0:
                    row_y += (rr - 0) * 0  # no-op clarity

                for cc in range(max_cols):
                    x0 = outer_pad_x + cc * (tile_w + hgap_images_px)

                    # Blank tile
                    tile = Image.new("L", (tile_w, tile_h), 255)

                    # only render in first d["n_cols"] columns
                    if cc < int(d["n_cols"]) and pos < len(chosen):
                        idx = chosen[pos]
                        im_np = _pad_to_hw_uint8(imgs_by_idx[idx], max_h, max_w, fill=128)
                        im_pil = Image.fromarray(im_np, mode="L")
                        tile.paste(im_pil, (0, 0))

                        # overlay event index in upper-right
                        sidx = str(idx)
                        tw, th = _text_wh(ddraw, sidx, font_event)
                        tx = max(0, tile_w - event_inset_x_px - tw)
                        ty = max(0, event_inset_y_px)

                        if event_text_bg:
                            bx0 = max(0, tx - event_text_bg_pad_px)
                            by0 = max(0, ty - event_text_bg_pad_px)
                            bx1 = min(tile_w, tx + tw + event_text_bg_pad_px)
                            by1 = min(tile_h, ty + th + event_text_bg_pad_px)
                            # light background rectangle for readability
                            ImageDraw.Draw(tile).rectangle([bx0, by0, bx1, by1], fill=255)

                        ImageDraw.Draw(tile).text((tx, ty), sidx, font=font_event, fill=0)

                        pos += 1

                    canvas.paste(tile, (x0, row_y))

            y += grid_h
            y += 2 * vgap_rows_px  # class gap

        try:
            canvas.save(out_png)
            return out_png
        except Exception as e:
            print(f"[FAIL] Could not save {out_png}: {e}")
            return None




############################################################################
#
# helper functions not callable from outside
#
############################################################################

def _ds_get_rtdc_path(ds) -> str:
    """Best-effort: get the backing RTDC path from an open dclab dataset."""
    for attr in ("path", "filename"):
        p = getattr(ds, attr, None)
        if isinstance(p, str) and p:
            return p
    # h5py-backed dataset
    try:
        p = ds.h5file.filename  # type: ignore[attr-defined]
        if isinstance(p, str) and p:
            return p
    except Exception:
        pass
    raise ValueError(
        "Could not determine RTDC file path from dataset. "
        "Please open ds via dclab.new_dataset(<path_to_rtdc>)."
    )



_CLASSIFICATION_META_KEY = "rtdc_tools_classification_json"
CLASSIFICATION_METADATA_VERSION = "2.0"


def _json_safe_class_map(class_map: Dict[Any, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, value in dict(class_map or {}).items():
        try:
            key = str(int(key))
        except Exception:
            key = str(key)
        out[key] = str(value)
    return out


def _restore_class_map(class_map: Dict[Any, Any]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for key, value in dict(class_map or {}).items():
        try:
            out[int(key)] = str(value)
        except Exception:
            continue
    return out


def _new_classification_registry() -> dict:
    return {"version": CLASSIFICATION_METADATA_VERSION, "entries": {}}


def _load_classification_registry_from_user(user_meta: Optional[dict]) -> dict:
    """Parse metadata v2.0, with a best-effort conversion of old single entries."""
    registry = _new_classification_registry()
    if not isinstance(user_meta, dict):
        return registry

    raw = user_meta.get(_CLASSIFICATION_META_KEY)
    if raw in (None, ""):
        return registry

    try:
        payload = json.loads(raw) if isinstance(raw, str) else copy.deepcopy(raw)
    except Exception:
        return registry
    if not isinstance(payload, dict):
        return registry

    if str(payload.get("version")) == CLASSIFICATION_METADATA_VERSION and isinstance(payload.get("entries"), dict):
        registry["entries"] = copy.deepcopy(payload["entries"])
        return registry

    # Best-effort import of metadata format 1.x.
    if any(k in payload for k in ("class_feature", "class_map", "classifier")):
        class_feature = str(payload.get("class_feature") or "unknown_class_feature")
        classifier = payload.get("classifier", {})
        if not isinstance(classifier, dict):
            classifier = {}
        registry["entries"][class_feature] = {
            "class_feature": class_feature,
            "probability_feature": payload.get("probability_feature"),
            "class_map": _json_safe_class_map(payload.get("class_map", {})),
            "classifier": classifier,
            "created_utc": None,
            "source_metadata_version": str(payload.get("version", "1.x")),
        }
    return registry


def _store_classification_registry_in_user(user_meta: dict, registry: dict) -> None:
    user_meta[_CLASSIFICATION_META_KEY] = json.dumps(registry, ensure_ascii=False)


def _register_classifier_metadata(
    meta: dict,
    *,
    class_feature: str,
    probability_feature: Optional[str],
    class_map: Dict[Any, Any],
    classifier: Dict[str, Any],
) -> dict:
    """Add one classifier entry, replacing an older entry with the same name."""
    meta.setdefault("user", {})
    registry = _load_classification_registry_from_user(meta["user"])

    classifier = copy.deepcopy(classifier) if isinstance(classifier, dict) else {}
    classifier_name = str(classifier.get("name", "")).strip()
    if not classifier_name:
        raise ValueError("classifier['name'] must be a non-empty string")

    # A classifier name is the public identifier, so keep it unique.
    for key, existing in list(registry.get("entries", {}).items()):
        if not isinstance(existing, dict):
            continue
        existing_classifier = existing.get("classifier", {})
        if isinstance(existing_classifier, dict) and str(existing_classifier.get("name", "")) == classifier_name:
            del registry["entries"][key]

    cf = str(class_feature)
    pf = None if probability_feature in (None, "") else str(probability_feature)
    registry["version"] = CLASSIFICATION_METADATA_VERSION
    registry.setdefault("entries", {})[cf] = {
        "class_feature": cf,
        "probability_feature": pf,
        "class_map": _json_safe_class_map(class_map),
        "classifier": classifier,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    _store_classification_registry_in_user(meta["user"], registry)
    return meta


def _make_classification_meta_json(
    class_feature: str,
    class_map: Dict[int, str],
    classifier: Dict[str, Any],
    probability_feature: Optional[str] = None,
) -> str:
    """Build a complete metadata-v2 registry containing one classifier."""
    meta = {"user": {}}
    _register_classifier_metadata(
        meta,
        class_feature=class_feature,
        probability_feature=probability_feature,
        class_map=class_map,
        classifier=classifier,
    )
    return meta["user"][_CLASSIFICATION_META_KEY]


def extract_rtdc_classifier_metadata(path_or_ds: Any) -> Dict[str, Any]:
    """Return a meaningful summary of every classifier registered in an RTDC file."""
    if isinstance(path_or_ds, (str, os.PathLike, Path)):
        with dclab.new_dataset(str(path_or_ds)) as ds:
            return extract_rtdc_classifier_metadata(ds)

    ds = path_or_ds
    meta = ds.config.as_dict(pop_filtering=True) or {}
    registry = _load_classification_registry_from_user(meta.get("user", {}))
    result: Dict[str, Any] = {
        "metadata_key": _CLASSIFICATION_META_KEY,
        "version": registry.get("version"),
        "n_classifiers": 0,
        "classifiers": [],
    }

    for key, raw_entry in registry.get("entries", {}).items():
        if not isinstance(raw_entry, dict):
            continue
        classifier = raw_entry.get("classifier", {})
        if not isinstance(classifier, dict):
            classifier = {}
        class_map = _restore_class_map(raw_entry.get("class_map", {}))
        result["classifiers"].append({
            "classifier_name": str(classifier.get("name", "")),
            "class_feature": str(raw_entry.get("class_feature", key)),
            "probability_feature": raw_entry.get("probability_feature"),
            "class_map": class_map,
            "n_classes": len(class_map),
            "created_utc": raw_entry.get("created_utc"),
            "classifier": copy.deepcopy(classifier),
        })

    result["n_classifiers"] = len(result["classifiers"])
    return result



def get_classifier_metadata(
    path_or_ds: Any,
    classifier_name: str = "default",
) -> Dict[str, Any]:
    """Return metadata for one classifier.

    ``classifier_name="default"`` selects the first classifier in metadata
    insertion order.  A concrete name selects that classifier explicitly.
    """
    info = extract_rtdc_classifier_metadata(path_or_ds)
    classifiers = info["classifiers"]
    if not classifiers:
        raise KeyError("No classifier metadata is stored in this RTDC dataset")

    wanted = str(classifier_name)
    if wanted == "default":
        return classifiers[0]

    matches = [item for item in classifiers if item["classifier_name"] == wanted]
    if not matches:
        available = [item["classifier_name"] for item in classifiers]
        raise KeyError(f"Classifier '{wanted}' not found. Available classifiers: {available}")
    if len(matches) > 1:
        raise ValueError(f"Classifier name '{wanted}' is not unique in the RTDC metadata")
    return matches[0]

def print_rtdc_classifier_metadata(path_or_ds: Any) -> Dict[str, Any]:
    """Print classifier metadata in a concise human-readable form and return it."""
    info = extract_rtdc_classifier_metadata(path_or_ds)
    print(f"Classification metadata version: {info['version']}")
    print(f"Number of classifiers: {info['n_classifiers']}")
    if not info["classifiers"]:
        print("No classifier metadata found.")
        return info

    for item in info["classifiers"]:
        print(f"\nClassifier: {item['classifier_name']}")
        print(f"  class feature:       {item['class_feature']}")
        pf = item["probability_feature"] if item["probability_feature"] is not None else "None (no probability filtering)"
        print(f"  probability feature: {pf}")
        classifier = item.get("classifier", {})
        for key in ("cnn_backbone", "model_file", "model_input_height", "model_input_width", "method", "publication"):
            if classifier.get(key) is not None:
                print(f"  {key}: {classifier[key]}")
        print("  classes:")
        for value, name in sorted(item["class_map"].items()):
            print(f"    {value}: {name}")
    return info


def _resolve_class_values(class_map: Dict[int, str], classes: Any) -> List[int]:
    if isinstance(classes, (str, int, np.integer)):
        requested = [classes]
    else:
        requested = list(classes)

    name_to_values: Dict[str, List[int]] = {}
    for value, name in class_map.items():
        name_to_values.setdefault(str(name), []).append(int(value))
        name_to_values.setdefault(str(name).lower(), []).append(int(value))

    values: List[int] = []
    for item in requested:
        if isinstance(item, (int, np.integer)):
            value = int(item)
            if value not in class_map:
                raise KeyError(f"Class value {value} is not present in the classifier class map")
            values.append(value)
        else:
            key = str(item)
            matches = name_to_values.get(key, name_to_values.get(key.lower(), []))
            if not matches:
                raise KeyError(f"Class name '{item}' is not present in the classifier class map")
            values.extend(matches)
    return sorted(set(values))


def make_classifier_mask(
    path_or_ds: Any,
    classifier_name: str,
    classes: Any,
    probability_threshold: Optional[float] = None,
    probability_comparison: str = ">",
) -> np.ndarray:
    """
    Return a Boolean event mask for selected classes of one classifier.

    `classes` may be a class number, class name, or a sequence of either. If the
    classifier has no probability feature, the probability condition is ignored
    and every event in the selected classes is retained.
    """
    if isinstance(path_or_ds, (str, os.PathLike, Path)):
        with dclab.new_dataset(str(path_or_ds)) as ds:
            return make_classifier_mask(
                ds,
                classifier_name=classifier_name,
                classes=classes,
                probability_threshold=probability_threshold,
                probability_comparison=probability_comparison,
            )

    ds = path_or_ds
    info = get_classifier_metadata(ds, classifier_name)
    class_feature = info["class_feature"]
    if class_feature not in ds.features:
        raise KeyError(f"Class feature '{class_feature}' is missing from the RTDC dataset")

    values = _resolve_class_values(info["class_map"], classes)
    labels = np.asarray(ds[class_feature], dtype=int)
    mask = np.isin(labels, values)

    probability_feature = info["probability_feature"]
    if probability_feature is None or probability_threshold is None:
        return mask
    if probability_feature not in ds.features:
        raise KeyError(f"Probability feature '{probability_feature}' is missing from the RTDC dataset")

    prob = np.asarray(ds[probability_feature], dtype=float)
    threshold = float(probability_threshold)
    operators = {
        ">": np.greater,
        ">=": np.greater_equal,
        "<": np.less,
        "<=": np.less_equal,
        "==": np.equal,
    }
    if probability_comparison not in operators:
        raise ValueError("probability_comparison must be one of: >, >=, <, <=, ==")
    return mask & np.isfinite(prob) & operators[probability_comparison](prob, threshold)


def _load_classification_meta_from_user(user_meta: dict) -> Tuple[Optional[str], Dict[int, str], Dict[str, Any]]:
    """
    Compatibility helper for code expecting one classifier.

    Returns a result only when exactly one classifier is registered; otherwise
    it returns `(None, {}, {})` because selecting by classifier name is required.
    """
    registry = _load_classification_registry_from_user(user_meta)
    entries = list(registry.get("entries", {}).values())
    if len(entries) != 1 or not isinstance(entries[0], dict):
        return None, {}, {}
    entry = entries[0]
    classifier = entry.get("classifier", {})
    if not isinstance(classifier, dict):
        classifier = {}
    return (
        str(entry.get("class_feature")) if entry.get("class_feature") is not None else None,
        _restore_class_map(entry.get("class_map", {})),
        classifier,
    )


def _allocate_userdef_features(path_or_ds: Any, count: int) -> List[str]:
    """Return the next unused `userdefN` feature names."""
    if isinstance(path_or_ds, (str, os.PathLike, Path)):
        with dclab.new_dataset(str(path_or_ds)) as ds:
            return _allocate_userdef_features(ds, count)

    ds = path_or_ds
    used = {str(name) for name in ds.features}
    try:
        info = extract_rtdc_classifier_metadata(ds)
        for item in info["classifiers"]:
            used.add(str(item["class_feature"]))
            if item["probability_feature"] is not None:
                used.add(str(item["probability_feature"]))
    except Exception:
        pass

    result: List[str] = []
    i = 1
    while len(result) < int(count):
        name = f"userdef{i}"
        if name not in used:
            result.append(name)
            used.add(name)
        i += 1
        if i > 10000:
            raise RuntimeError("Could not allocate free userdef feature names")
    return result


def _find_true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return list of (start, stop) for contiguous True runs (stop is exclusive)."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    # pad with False at both ends to detect transitions
    m = np.concatenate([[False], mask, [False]])
    starts = np.flatnonzero((~m[:-1]) & (m[1:]))
    ends = np.flatnonzero((m[:-1]) & (~m[1:]))
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _split_run(start: int, stop: int, chunk: int) -> List[Tuple[int, int]]:
    out = []
    i = int(start)
    stop = int(stop)
    chunk = max(1, int(chunk))
    while i < stop:
        j = min(i + chunk, stop)
        out.append((i, j))
        i = j
    return out


def _strip_basin_metadata(meta: dict) -> dict:
    """Remove basin definitions from a copied config dict (we add our own basin later)."""
    if not isinstance(meta, dict):
        return meta
    meta = dict(meta)  # shallow copy
    meta.pop("basins", None)
    meta.pop("basin", None)
    return meta


def _deep_merge_dicts(base: dict, incoming: dict) -> dict:
    """Recursively merge two dictionaries (incoming overwrites base)."""
    if not isinstance(base, dict):
        base = {}
    if not isinstance(incoming, dict):
        return dict(base)
    out = dict(base)
    for k, v in incoming.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dicts(out[k], v)
        else:
            out[k] = v
    return out



def _infer_int_class_feature_name(*args, **kwargs):
    """Deprecated metadata-v1 helper; retained as a no-op."""
    return None



def _get_ultimate_basin_locs_and_format(basin_rtdc_path: str) -> Tuple[List[str], str]:
    """
    Determine ultimate basin locations of `basin_rtdc_path` (avoid chaining referrers).
    Returns (basin_locs, basin_format).
    """
    with dclab.new_dataset(basin_rtdc_path) as ds_in:
        cfg = ds_in.config.as_dict(pop_filtering=True)
    basins = cfg.get("basins", None) if isinstance(cfg, dict) else None
    if isinstance(basins, dict) and len(basins) > 0:
        first = next(iter(basins.values()))
        if isinstance(first, dict):
            locs = list(first.get("locs", [basin_rtdc_path]))
            fmt = str(first.get("format", "hdf5"))
            return locs, fmt
    return [basin_rtdc_path], "hdf5"



def _update_classification_meta_class_feature(*args, **kwargs):
    """Deprecated metadata-v1 helper; retained as a no-op."""
    return None




############################################################################
#
# common functions for preprocessing images of rtdc events before use in ML training and classification
#
############################################################################




# =============================================================================
# IMAGE-MODEL LAZY WRAPPERS
# =============================================================================



def classify_ds_images_ml(
    ds: Any,
    mask: Optional[Sequence[bool]] = None,
    *,
    path_to_model: str,
    preprocessing_function: Optional[Callable[..., np.ndarray]] = None,
    classifier_name: Optional[str] = None,
    mask_description: Optional[str] = None,
    verbose: bool = True,
    show_progress: bool = True,
    extra_pixels: int = 15,
    batch_size: int = 1024,
    num_workers: int = 0,
    prefetch_batches: int = 8,
    preprocess_chunk_size: int = 256,
    use_amp: Optional[bool] = None,
    preprocess_kwargs: Optional[dict] = None,
    cv_threads_per_worker: int = 2,
    exclude_x_overlapping_events: bool = True,
    min_x_overlap: float = 0.0,
    min_x_overlap_fraction: float = 0.0,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Lazy wrapper for image-model inference on an open RTDC dataset."""
    from rtdc_toolbox_ml import classify_ds_images_ml as implementation

    return implementation(
        ds,
        mask=mask,
        path_to_model=path_to_model,
        preprocessing_function=preprocessing_function,
        classifier_name=classifier_name,
        mask_description=mask_description,
        verbose=verbose,
        show_progress=show_progress,
        extra_pixels=extra_pixels,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_batches=prefetch_batches,
        preprocess_chunk_size=preprocess_chunk_size,
        use_amp=use_amp,
        preprocess_kwargs=preprocess_kwargs,
        cv_threads_per_worker=cv_threads_per_worker,
        exclude_x_overlapping_events=exclude_x_overlapping_events,
        min_x_overlap=min_x_overlap,
        min_x_overlap_fraction=min_x_overlap_fraction,
    )



def classify_rtdc_images_ml(
    path_to_rtdc: str,
    *,
    path_to_model: str,
    preprocessing_function: Optional[Callable[..., np.ndarray]] = None,
    classifier_name: Optional[str] = None,
    mask: Optional[Sequence[bool]] = None,
    mask_description: Optional[str] = None,
    output_suffix: str = "_images_ml",
    overwrite: bool = False,
    verbose: bool = True,
    show_progress: bool = True,
    extra_pixels: int = 15,
    batch_size: int = 1024,
    num_workers: int = 0,
    prefetch_batches: int = 8,
    preprocess_chunk_size: int = 256,
    use_amp: Optional[bool] = None,
    preprocess_kwargs: Optional[dict] = None,
    cv_threads_per_worker: int = 2,
    exclude_x_overlapping_events: bool = True,
    min_x_overlap: float = 0.0,
    min_x_overlap_fraction: float = 0.0,
) -> str:
    """Lazy wrapper for image-model classification of an RTDC file."""
    from rtdc_toolbox_ml import classify_rtdc_images_ml as implementation

    return implementation(
        path_to_rtdc=path_to_rtdc,
        path_to_model=path_to_model,
        preprocessing_function=preprocessing_function,
        classifier_name=classifier_name,
        mask=mask,
        mask_description=mask_description,
        output_suffix=output_suffix,
        overwrite=overwrite,
        verbose=verbose,
        show_progress=show_progress,
        extra_pixels=extra_pixels,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_batches=prefetch_batches,
        preprocess_chunk_size=preprocess_chunk_size,
        use_amp=use_amp,
        preprocess_kwargs=preprocess_kwargs,
        cv_threads_per_worker=cv_threads_per_worker,
        exclude_x_overlapping_events=exclude_x_overlapping_events,
        min_x_overlap=min_x_overlap,
        min_x_overlap_fraction=min_x_overlap_fraction,
    )


def train_model_on_images(
    path_to_training_set: str,
    training_results_dir: str,
    model_name: str,
    zip: bool = True,
    cnn_backbone: str = "resnet18",
    img_ext: str = ".png",
    n_epochs: int = 15,
    W0: int = 160,
    val_split: float = 0.2,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    brightness_jitter: float = 0.0,
    contrast_jitter: float = 0.0,
    imbalance_cfg: Optional[Any] = None,
    seed: int = 123,
    num_workers: int = 0,
    pretrained: bool = True,
    cache_in_ram: bool = False,
) -> str:
    """Train an image-only CNN from preprocessed grayscale images."""
    from rtdc_toolbox_ml import train_model_on_images as implementation
    return implementation(
        path_to_training_set=path_to_training_set,
        training_results_dir=training_results_dir,
        model_name=model_name,
        zip=zip,
        cnn_backbone=cnn_backbone,
        img_ext=img_ext,
        n_epochs=n_epochs,
        W0=W0,
        val_split=val_split,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        brightness_jitter=brightness_jitter,
        contrast_jitter=contrast_jitter,
        imbalance_cfg=imbalance_cfg,
        seed=seed,
        num_workers=num_workers,
        pretrained=pretrained,
        cache_in_ram=cache_in_ram,
    )


def load_image_model(model_path: str):
    """Load an image-only CNN model bundle saved by train_model_on_images."""
    from rtdc_toolbox_ml import load_image_model as implementation
    return implementation(model_path)


