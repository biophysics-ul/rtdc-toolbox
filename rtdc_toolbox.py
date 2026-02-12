"""
rtdc_toolbox (core)

Torch-free utilities for working with RT-DC (.rtdc) files:
- image export and preprocessing
- RT-DC stability analysis report
- WBC classification (Kaliman 2025, scikit-learn based)
- RTDC metadata helpers + writing new RTDC files with basin reference

ML (PyTorch/torchvision) functionality is implemented in `rtdc_toolbox_ml.py`.
This core module provides thin wrappers (lazy imports) so existing scripts that
import ML functions from `rtdc_toolbox` keep working, as long as torch is installed.
"""
from __future__ import annotations

import os
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import cv2 as cv

from tqdm import tqdm
import random
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

import dclab


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

    The preprocessing in this fucntion comprises:
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
# Nest we define the main toolbox functions
#
###########################################################################

def get_classes_from_rtdc(path_to_rtdc: str) -> Dict[int, str]:
    """
    Read class mapping from unified RTDC metadata.

    This toolbox uses a single convention:
      ds.config["user"][_CLASSIFICATION_META_KEY] is a JSON string/dict with:
        - "class_map": { "<int_as_str>": "<class_name>", ... }

    Returns
    -------
    dict[int, str]
        Example: {0: "unclassified", 1: "WBC", ...}
        Returns {} if classification metadata is missing or malformed.
    """
    with dclab.new_dataset(path_to_rtdc) as ds:
        meta = ds.config.as_dict(pop_filtering=True) or {}
        user_meta = meta.get("user", {}) if isinstance(meta, dict) else {}
        _, class_names, _ = _load_classification_meta_from_user(user_meta)
    return dict(sorted(class_names.items(), key=lambda kv: kv[0]))

def extract_preprocessed_images_from_rtdc_to_zip(
    rtdc_path: Union[str, Path],
    zip_path: Union[str, Path] = "export.zip",
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None, # export only images in mask (if None, export all images)
    extra_pixels: int = 15, # extra pixels left/right from the cell to include
    no_overlap: bool = True, # prevent overlaping with adjacent cells
    img_ext: str = ".png", # the type of the exported images
    n: int = 1000, # number of images to export
    random_images: bool = False # export first n images or n random images
) -> None:
    """
    Extract preprocessed images from an .rtdc into a ZIP using OpenCV encoding.
    - If mask is None: all events are exported.
    - Else: mask must be a 1/0 or True/False array with length == len(ds).
    - preprocess_rtdc_image(ds, i, extra_pixels=..., w=...) must exist.
    - Filenames inside ZIP: "<event><ext>".
    """
    with dclab.new_dataset(rtdc_path) as ds:

        if mask is None:
            idxs = np.arange(len(ds), dtype=int)
        else:
            mask = np.asarray(mask).astype(bool)
            if len(mask) != len(ds):
                print(f"ERROR: mask length {len(mask)} != number of events {len(ds)}")
                return
            idxs = np.where(mask)[0].astype(int)
            if len(idxs) == 0:
                print("No events selected (mask contains no True/1 values).")
                return

        if random_images and len(idxs) > n:
            idxs = np.asarray(random.sample(list(idxs), k=int(n)), dtype=int)
        else:
            idxs = idxs[: int(n)]

        ext = img_ext if img_ext.startswith(".") else "." + img_ext

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED, allowZip64=True) as zipf:
            print("Extracting images from rtdc to zip.")

            for i in tqdm(idxs):
                img = preprocess_rtdc_image(ds, int(i), extra_pixels=extra_pixels,no_overlap=no_overlap)
                if img is None:
                    continue

                ok, buf = cv.imencode(ext, img)
                if not ok:
                    print(f"WARNING: cv.imencode failed for event {i}")
                    continue

                filename = f"{i}{ext}"
                zipf.writestr(filename, buf.tobytes())

        print(f"Done. Wrote {len(idxs)} images to: {zip_path}")

def save_features_to_rtdc(
    basin_rtdc_path: str,
    output_rtdc_path: str,
    feature_dict: Dict[str, np.ndarray],
    metadata: dict,
    overwrite: bool = True,
) -> str:
    """
    Create a new RTDC file at `output_rtdc_path` that points to `basin_rtdc_path` (or its ultimate basin),
    and stores in it the features from `feature_dict` plus `metadata`.

    Parameters
    ----------
    basin_rtdc_path : str
        Source RTDC to link to.
    output_rtdc_path : str
        New RTDC to write.
    feature_dict : dict
        {feature_name: np.ndarray} arrays must be length N of the basin dataset.
    metadata : dict
        Metadata dict as produced by ds.config.as_dict(pop_filtering=True), with extra user info.
    overwrite : bool
        If False and output exists -> raise.

    Returns
    -------
    output_rtdc_path : str
    """
    out_path = str(output_rtdc_path)
    if os.path.exists(out_path) and not overwrite:
        print(f"[SKIP] file aready exists (you can set overwrite to True): {out_path}", flush=True)
        return out_path

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    basin_locs, basin_format = _get_ultimate_basin_locs_and_format(str(basin_rtdc_path))

    # Start from the basin dataset metadata and merge user-provided metadata on top.
    with dclab.new_dataset(basin_rtdc_path) as ds0:
        base_meta = _strip_basin_metadata(ds0.config.as_dict(pop_filtering=True))
        n = len(ds0)

    incoming_meta = _strip_basin_metadata(metadata if isinstance(metadata, dict) else {})
    meta = _deep_merge_dicts(base_meta, incoming_meta)

    # If the caller provides unified classification metadata, ensure it points to the
    # integer feature we store class labels into.
    class_feature_name = _infer_int_class_feature_name(feature_dict)
    if class_feature_name:
        _update_classification_meta_class_feature(meta, class_feature_name)
        meta.setdefault("user", {})
        # For WBC Kaliman output, also store the feature name explicitly.

    # Validate feature lengths
    for k, v in dict(feature_dict).items():
        arr = np.asarray(v)
        if arr.shape[0] != n:
            raise ValueError(f"Feature '{k}' length {arr.shape[0]} != number of events {n}")

    # First write metadata + features
    with dclab.RTDCWriter(out_path, mode="reset") as hw:
        hw.store_metadata(meta)
        for k, v in dict(feature_dict).items():
            hw.store_feature(str(k), np.asarray(v))

    # Then attach basin reference to the *ultimate* source file(s)
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

def classify_rtdc_WBC_klmn25(
    path_to_rtdc: str,
    class_feature: str = "userdef1",
    echo: bool = True,
    overwrite: bool = False,
    out_ext: str = "_klmn25",
) -> str:
    """
    Classifies RT-DC events according to:
      Kaliman et al., Biophysical Journal, 2025, DOI: 10.1016/j.bpj.2025.10.007
      "Automation and improvement of WBC mechanical profiling in deformability cytometry"

    Writes a *new* .rtdc file that:
      - is linked to the original file via a basin reference ("original")
      - contains integer class labels stored in `class_feature`

    Returns
    -------
    out_path : str
        Path to the newly written RTDC file.
    """
    # ------------------- Output path -------------------
    if str(path_to_rtdc).lower().endswith(".rtdc"):
        out_path = str(path_to_rtdc)[:-5] + str(out_ext) + ".rtdc"
    else:
        out_path = str(path_to_rtdc) + str(out_ext) + ".rtdc"

    if os.path.exists(out_path) and not overwrite:
        print(f"[SKIP] file aready exists (you can set overwrite to True): {out_path}", flush=True)
        return out_path

    with dclab.new_dataset(path_to_rtdc) as ds:
        labels, meta = classify_ds_WBC_klmn25(ds, mask=None, echo=echo)

    # Store the classification result in a new RTDC file linked to the input dataset.
    save_features_to_rtdc(
        basin_rtdc_path=str(path_to_rtdc),
        output_rtdc_path=str(out_path),
        feature_dict={str(class_feature): labels},
        metadata=meta,
        overwrite=True,
    )

    return out_path



def classify_ds_WBC_klmn25(
    ds,
    mask: Optional[np.ndarray] = None,
    *,
    echo: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Kaliman 2025 WBC classification on an already-open dataset `ds`,
    optionally only for events where `mask == True`.

    Returns:
        labels (np.uint8, len N),
        meta   (dict) metadata describing the classifier and class names

    Notes:
      - Events outside `mask` are set to class=0 ("unclassified").
      - If mask is None, all events are classified.
    """

    # Step 1 thresholds
    AREA_um_MIN = 30.0
    AREA_um_MAX = 100.0
    DEFORM_MAX = 0.12
    AREA_RATIO_MAX = 1.10

    # Step 2b brightness p90 window
    BRIGHT_P90_MIN = -10.0
    BRIGHT_P90_MAX = 15.0

    # Step 2c deformation window
    DEFORM_RAW_MIN = 0.06
    DEFORM_RAW_MAX = 0.26

    TEX_CONTRAST_MAX = 250.0
    RND_SEED = 42

    DC_EVENT_CLASSES = {
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

    N = len(ds)
    if mask is None:
        mask_arr = np.ones(N, dtype=bool)
    else:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape[0] != N:
            raise ValueError(f"mask length {mask_arr.shape[0]} != number of events {N}")

    labels = np.zeros(N, dtype=np.uint8)

    if echo:
        print(f"Total events: {N:,}", flush=True)
        if mask is not None:
            print(f"Mask True events: {int(mask_arr.sum()):,}", flush=True)

    if not np.any(mask_arr):
        meta = ds.config.as_dict(pop_filtering=True)
        meta.setdefault("user", {})
        inv = {0: "unclassified"}
        meta["user"][_CLASSIFICATION_META_KEY] = _make_classification_meta_json(
            class_feature="",
            class_map=inv,
            classifier={"name": "Kaliman2025"},
        )
        return labels, meta

    # Pull features (full arrays)
    required = [
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
    missing = [f for f in required if f not in ds.features]
    if missing:
        raise KeyError(
            "Kaliman 2025 WBC classifier requires these missing features in the RTDC file: "
            + ", ".join(missing)
        )

    area_um = ds["area_um"][:]
    area_um_raw = ds["area_um_raw"][:]
    area_ratio = ds["area_ratio"][:]
    deform = ds["deform"][:]
    deform_raw = ds["deform_raw"][:]
    bright_p90 = ds["bright_perc_90"][:]
    bright_avg = ds["bright_avg"][:]
    tex_con_avg = ds["tex_con_avg"][:]
    tex_den_avg = ds["tex_den_avg"][:]
    tex_sva_avg = ds["tex_sva_avg"][:]
    tex_idm_ptp = ds["tex_idm_ptp"][:]
    tex_den_ptp = ds["tex_den_ptp"][:]

    # Build local-to-global mapping; we will compute boolean masks full-length but only for selected events
    sel = mask_arr  # alias

    # Initialize inside-mask labels as Kaliman unclassified (0)
    labels_in = np.full(N, DC_EVENT_CLASSES["unclassified"], dtype=np.int32)

    # STEP 1
    crude_wbcs = sel & (
        (area_um >= AREA_um_MIN) &
        (area_um <= AREA_um_MAX) &
        (deform <= DEFORM_MAX) &
        (area_ratio <= AREA_RATIO_MAX)
    )
    step1_discarded = sel & (~crude_wbcs)
    labels_in[step1_discarded] = DC_EVENT_CLASSES["step1_discarded"]
    if echo:
        print(f"Step 1 kept: {int(crude_wbcs.sum()):,} (discarded: {int(step1_discarded.sum()):,})", flush=True)

    # STEP 2a
    base2 = crude_wbcs
    idx2 = np.flatnonzero(base2)
    if idx2.size >= 2:
        data2a = np.column_stack((tex_idm_ptp[base2], tex_den_ptp[base2]))
        gmm2a = GaussianMixture(n_components=2, random_state=RND_SEED).fit(data2a)
        pred2a = gmm2a.predict(data2a)
        centers2a = gmm2a.means_
        norm2 = np.sum(centers2a ** 2, axis=1)
        rbc_cluster = int(np.argmax(norm2))
        wbc_cluster = 1 - rbc_cluster

        wbc_step2a = np.zeros(N, dtype=bool)
        wbc_step2a[idx2] = (pred2a == wbc_cluster)
    elif idx2.size == 1:
        # Too few events for a 2-component GMM: keep the single event.
        wbc_step2a = base2.copy()
    else:
        wbc_step2a = np.zeros(N, dtype=bool)

    if echo:
        print(f"Step 2a (GMM) kept: {int(wbc_step2a.sum()):,}")

    # STEP 2b
    wbc_step2b = wbc_step2a & (bright_p90 >= BRIGHT_P90_MIN) & (bright_p90 <= BRIGHT_P90_MAX)
    if echo:
        print(f"Step 2b (brightness p90) kept: {int(wbc_step2b.sum()):,}")

    # STEP 2c
    wbcs = wbc_step2b & (deform_raw >= DEFORM_RAW_MIN) & (deform_raw <= DEFORM_RAW_MAX)
    if echo:
        print(f"Step 2c (deformation raw) kept: {int(wbcs.sum()):,}")

    step2_discarded = base2 & (~wbcs)
    labels_in[step2_discarded] = DC_EVENT_CLASSES["step2_discarded"]
    if echo:
        print(f"Step 2 kept: {int(wbcs.sum()):,} (discarded: {int(step2_discarded.sum()):,})")

    # STEP 3
    base3 = wbcs
    idx3 = np.flatnonzero(base3)
    crude_lymphos = np.zeros(N, dtype=bool)
    if idx3.size >= 2:
        data3 = np.column_stack((area_um[base3], area_um_raw[base3]))
        km3 = KMeans(n_clusters=2, random_state=RND_SEED, n_init="auto")
        pred3 = km3.fit_predict(data3)
        centers3 = km3.cluster_centers_
        crude_lymph_cluster = int(np.argmin(centers3.mean(axis=1)))
        crude_lymphos[idx3] = (pred3 == crude_lymph_cluster)
    elif idx3.size == 1:
        # Only one event: cannot split into two clusters; assign to lymph branch.
        crude_lymphos[idx3[0]] = True
    granulos_monos = base3 & (~crude_lymphos)

    if echo:
        print(f"Step 3 Crude Lymphos: {int(crude_lymphos.sum()):,}")
        print(f"Step 3 Granulos+Monos: {int(granulos_monos.sum()):,}")

    # STEP 4
    base4a = crude_lymphos
    step4a_accepted = base4a & (tex_con_avg <= TEX_CONTRAST_MAX)

    clean_lymphos = np.zeros(N, dtype=bool)
    idx4 = np.flatnonzero(step4a_accepted)
    if idx4.size >= 2:
        data4 = np.column_stack((bright_avg[step4a_accepted], tex_con_avg[step4a_accepted]))
        gmm4 = GaussianMixture(n_components=2, random_state=RND_SEED).fit(data4)
        pred4 = gmm4.predict(data4)
        centers4 = gmm4.means_
        clean_lymph_cluster = int(np.argmax(centers4[:, 0]))
        clean_lymphos[idx4] = (pred4 == clean_lymph_cluster)
    elif idx4.size == 1:
        # Too few events for a 2-component GMM: accept the single event.
        clean_lymphos[idx4[0]] = True
    lymphocytes = clean_lymphos
    labels_in[lymphocytes] = DC_EVENT_CLASSES["lymphocytes"]

    step4_discarded = base4a & (~lymphocytes)
    labels_in[step4_discarded] = DC_EVENT_CLASSES["step4_discarded"]
    if echo:
        print(f"Step 4: Lymphocytes: {int(lymphocytes.sum()):,} (discarded: {int(step4_discarded.sum()):,})")

    # STEP 5
    base5 = granulos_monos
    idx5 = np.flatnonzero(base5)
    neutrophils = np.zeros(N, dtype=bool)
    if idx5.size >= 3:
        data5 = np.column_stack((bright_avg[base5], tex_den_avg[base5], tex_sva_avg[base5]))
        gmm5 = GaussianMixture(n_components=3, random_state=RND_SEED).fit(data5)
        pred5 = gmm5.predict(data5)
        counts5 = np.bincount(pred5, minlength=3)
        neut_cluster = int(np.argmax(counts5))
        neutrophils[idx5] = (pred5 == neut_cluster)
    elif idx5.size > 0:
        # Too few events for a 3-component GMM: assign all to neutrophils.
        neutrophils[idx5] = True
    labels_in[neutrophils] = DC_EVENT_CLASSES["neutrophils"]
    monos_eos = base5 & (~neutrophils)

    if echo:
        print(f"Step 5: Neutrophils: {int(neutrophils.sum()):,}")
        print(f"Step 5: Remaining (Monos+Eos candidates): {int(monos_eos.sum()):,}")

    # STEP 6
    base6 = monos_eos
    monocytes = np.zeros(N, dtype=bool)
    eosinophils = np.zeros(N, dtype=bool)

    if neutrophils.any():
        neut_tex_den = tex_den_avg[neutrophils]
        neut_max = float(np.max(neut_tex_den))
        neut_min = float(np.min(neut_tex_den))
        eosinophils = base6 & (tex_den_avg > neut_max)
        monocytes = base6 & (tex_den_avg < neut_min)

    labels_in[monocytes] = DC_EVENT_CLASSES["monocytes"]
    labels_in[eosinophils] = DC_EVENT_CLASSES["eosinophils"]

    step6_discarded = base6 & (~monocytes) & (~eosinophils)
    labels_in[step6_discarded] = DC_EVENT_CLASSES["step6_discarded"]

    if echo:
        print(f"Step 6: Monocytes: {int(monocytes.sum()):,}")
        print(f"Step 6: Eosinophils: {int(eosinophils.sum()):,}")
        print(f"Step 6: Discarded: {int(step6_discarded.sum()):,}")

    # merge into output labels
    labels[mask_arr] = labels_in[mask_arr]

    # metadata
    meta = ds.config.as_dict(pop_filtering=True)
    meta.setdefault("user", {})

    inv = {int(v): str(k) for k, v in DC_EVENT_CLASSES.items()}
    inv[0] = "unclassified"
    meta["user"][_CLASSIFICATION_META_KEY] = _make_classification_meta_json(
        class_feature="",
        class_map=inv,
        classifier={"name": "Kaliman2025"},
    )

    return labels, meta


def analyze_rtdc(
    path_to_rtdc: Union[str, Path],
    class_feature: Optional[str] = "userdef1",
    features_to_plot: Sequence[str] = ("area_um", "deform"),
    n_window: int = 10000,
    step: int = 1000,
    out_png: Optional[Union[str, Path]] = None,
) -> Optional[Path]:

    """
    Analyze ONE RTDC file and save a PNG report in the same folder.
    The report comprises plots of N(t), deform(t) ... for classes defined in metadata
    If class_feature is None it plots data for the whole dataset

    Returns:
        Path to PNG (Path) on success, or None if input file missing / failed.
    """
    rtdc_path = Path(path_to_rtdc)
    if not rtdc_path.exists():
        print(f"[SKIP] RTDC not found: {rtdc_path}")
        return None

    features_to_plot = list(features_to_plot) 

    # output name: default "<stem>.png" (e.g. xxx.rtdc -> xxx.png)
    out_png = Path(out_png) if out_png is not None else rtdc_path.with_suffix(".png")

    # Read class names from RTDC metadata first (opens+closes file for clarity).
    class_names = get_classes_from_rtdc(str(rtdc_path))

    def get_time(ds):
        return np.asarray(ds["time"], float) if "time" in ds.features else np.arange(len(ds), dtype=float)

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

        # classes
        if class_feature is None:
            cls = np.zeros(len(ds), dtype=int)
            local_names = {0: "All events"}
        else:
            if class_feature not in ds.features:
                print(f"[SKIP] {rtdc_path.name}: class_feature '{class_feature}' not found")
                return None
            cls = np.asarray(ds[class_feature], dtype=int)
            local_names = class_names

        if len(cls) == 0:
            print(f"[SKIP] {rtdc_path.name}: no events")
            return None

        # feature arrays (missing -> NaN)
        feat_arr = {}
        for f in features_to_plot:
            feat_arr[f] = np.asarray(ds[f], float) if f in ds.features else np.full(len(cls), np.nan, float)

        win = slices(len(cls), n_window, step)
        t_centers = np.array([np.nanmean(t[s]) for s in win], float)

        classes = np.unique(cls)
        n_rows = len(classes)
        n_cols = 1 + len(features_to_plot)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, max(2.5, 1.9 * n_rows)), sharex=True)
        if n_rows == 1:
            axes = np.array([axes])

        for ri, c in enumerate(classes):
            cname = local_names.get(int(c), str(int(c)))

            # --- Column 0: percent
            ax = axes[ri, 0]
            y = np.array([100.0 * np.mean(cls[s] == c) for s in win], float)
            overall = 100.0 * np.mean(cls == c)
            ax.axhline(overall, color="red", lw=1, alpha=0.8)
            ax.text(0.98, 0.98, f"avg={overall:.2f}%", transform=ax.transAxes,
                    ha="right", va="top", color="red", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7))
            ax.plot(t_centers, y)
            if ri == 0:
                ax.set_title("% of events in window")
            ax.set_ylabel(f"{cname}\nPercent (%)")
            ax.grid(True, alpha=0.25)
            if ri == n_rows - 1:
                ax.set_xlabel("Time")

            # --- Feature columns: mean ± SD
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

                overall = np.nanmean(vals[cls == c]) if np.any(cls == c) else np.nan
                if np.isfinite(overall):
                    ax.axhline(overall, color="red", lw=1, alpha=0.8)
                    ax.text(0.98, 0.98, f"avg={overall:.3g}", transform=ax.transAxes,
                            ha="right", va="top", color="red", fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7))

                ax.plot(t_centers, means)
                if np.isfinite(sds).any():
                    ax.fill_between(t_centers, means - sds, means + sds, alpha=0.2)

                if ri == 0:
                    ax.set_title(f"Mean ± SD: {f}")
                ax.set_ylabel(f)
                ax.grid(True, alpha=0.25)
                if ri == n_rows - 1:
                    ax.set_xlabel("Time")

        fig.suptitle(f"{rtdc_path.name} | N total = {n_total_events} | class_feature={class_feature} | window={n_window}", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.98])

        try:
            fig.savefig(out_png, dpi=200)
            plt.close(fig)
            return out_png
        except Exception as e:
            plt.close(fig)
            print(f"[FAIL] Could not save {out_png}: {e}")
            return None

def export_examples_of_cells_for_classes_in_rtdc(
    path_to_rtdc: Union[str, Path],
    report_ext: str = "_report_ext",
    N_img: int = 30,
    class_feature: str = "userdef1",
    prob_feature: str = "userdef2",
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

    # Class names from metadata (fallback to int string)
    class_names_meta = get_classes_from_rtdc(str(rtdc_path))

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

        have_prob = (prob_feature is not None) and (str(prob_feature) in ds.features)
        if have_prob:
            prob = np.asarray(ds[prob_feature], dtype=float)
            finite = np.isfinite(prob)
            above = finite & (prob >= float(prob_thresh))
            below = finite & (prob < float(prob_thresh))
            sel_mask = above if keep_above_thresh else below
        else:
            above = np.zeros(n_total, dtype=bool)
            below = np.zeros(n_total, dtype=bool)
            sel_mask = np.ones(n_total, dtype=bool)

        # Select indices per class
        per_class = []
        max_cols = 1
        all_selected = []

        for c in classes:
            c = int(c)
            in_class = (cls == c)
            n_in_class = int(in_class.sum())

            if have_prob:
                n_above = int((in_class & above).sum())
                n_below = int((in_class & below).sum())
            else:
                n_above = n_in_class
                n_below = 0

            candidates = np.where(in_class & sel_mask)[0].astype(int)
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
                f"class feature='{class_feature}'"
                ]
            if have_prob:
                side = ">=" if keep_above_thresh else "<"
                lines += [
                    f"probability feature='{prob_feature}'",
                    f"probability threshold: {side} {prob_thresh:g}",
                ]
            else:
                lines.append(f"probability feature='{prob_feature}' missing (no threshold filter)")
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
                
                f" | N={n_total} | "
                f"class feature='{class_feature}' | probability feature='{prob_feature}' | "
                f"probability threshold: {side} {prob_thresh:g}"
            )
        else:
            title = (
                f"{rtdc_path.name} | N={n_total} | "
                f"class feature='{class_feature}' | probability feature='{prob_feature}' (missing; no threshold filter)"
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

            if have_prob:
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

def _make_classification_meta_json(
    class_feature: str,
    class_map: Dict[int, str],
    classifier: Dict[str, Any],
) -> str:
    """Build a JSON string for storing classification metadata in RTDC attrs."""
    payload = {
        "version": 1,
        "class_feature": str(class_feature),
        # store as {str(int): str} because JSON keys must be strings and HDF5 attrs are bytes/str
        "class_map": {str(int(k)): str(v) for k, v in dict(class_map).items()},
        "classifier": dict(classifier) if classifier is not None else {},
    }
    return json.dumps(payload, ensure_ascii=False)

def _load_classification_meta_from_user(user_meta: dict) -> Tuple[Optional[str], Dict[int, str], Dict[str, Any]]:
    """
    Extract (class_feature, class_names, classifier) from unified RTDC user metadata.

    Unified convention (NO fallbacks):
      user[_CLASSIFICATION_META_KEY] is a JSON string (or dict) with:
        - "version": int
        - "class_feature": str (may be empty; will be updated by save_features_to_rtdc)
        - "class_map": { "<int_as_str>": "<class_name>", ... }
        - "classifier": dict (optional)

    Returns
    -------
    class_feature : Optional[str]
    class_names   : dict[int, str]
    classifier    : dict[str, Any]
    """
    if not isinstance(user_meta, dict):
        return None, {}, {}

    raw = user_meta.get(_CLASSIFICATION_META_KEY, None)
    if raw is None:
        return None, {}, {}

    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return None, {}, {}

    if not isinstance(payload, dict):
        return None, {}, {}

    cf = payload.get("class_feature", None)
    cmap = payload.get("class_map", None)
    classifier = payload.get("classifier", {}) or {}

    class_names: Dict[int, str] = {}
    if isinstance(cmap, dict):
        for k, v in cmap.items():
            try:
                class_names[int(k)] = str(v)
            except Exception:
                continue

    if not isinstance(classifier, dict):
        classifier = {}

    return (str(cf) if cf is not None else None), class_names, classifier


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


def _infer_int_class_feature_name(feature_dict: Dict[str, np.ndarray]) -> Optional[str]:
    """Infer which feature key corresponds to integer class labels."""
    cands = []
    for k, v in dict(feature_dict).items():
        arr = np.asarray(v)
        if np.issubdtype(arr.dtype, np.integer):
            cands.append(str(k))
    if len(cands) == 1:
        return cands[0]
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


def _update_classification_meta_class_feature(meta: dict, class_feature: str) -> None:
    """If unified classification JSON exists, update its 'class_feature' field."""
    try:
        user = meta.get("user", {})
        if not isinstance(user, dict):
            return
        raw = user.get(_CLASSIFICATION_META_KEY, None)
        if not isinstance(raw, str) or not raw:
            return
        payload = json.loads(raw)
        if isinstance(payload, dict):
            payload["class_feature"] = str(class_feature)
            user[_CLASSIFICATION_META_KEY] = json.dumps(payload, ensure_ascii=False)
            meta["user"] = user
    except Exception:
        pass



############################################################################
#
# common functions for preprocessing images of rtdc events before use in ML training and classification
#
############################################################################


# -------------------------------------------------------------------------
# Lazy wrappers for backward compatibility (ML lives in rtdc_toolbox_ml.py)
# -------------------------------------------------------------------------

def classify_rtdc_ml(*args, **kwargs):
    """Lazy wrapper for ML classification (requires torch)."""
    from rtdc_toolbox_ml import classify_rtdc_ml as _impl
    return _impl(*args, **kwargs)

def train_model(*args, **kwargs):
    """Lazy wrapper for ML training (requires torch)."""
    from rtdc_toolbox_ml import train_model as _impl
    return _impl(*args, **kwargs)

def load_model(*args, **kwargs):
    """Lazy wrapper for model loading (requires torch)."""
    from rtdc_toolbox_ml import load_model as _impl
    return _impl(*args, **kwargs)