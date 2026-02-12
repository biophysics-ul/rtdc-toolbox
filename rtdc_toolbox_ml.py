"""
rtdc_toolbox_ml

PyTorch/torchvision-based ML training and classification for RT-DC images.

This module depends on torch/torchvision.
It imports the torch-free core utilities from `rtdc_toolbox`.
"""
from __future__ import annotations

import os
import json
import time
import copy
import zipfile
import datetime
import contextlib
import multiprocessing as mp
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2 as cv
from PIL import Image

from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import dclab

# Import core utilities (torch-free)
import rtdc_toolbox as core

# Torch/torchvision are REQUIRED for this module
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as tvm

_TORCH_AVAILABLE = True
_TORCH_IMPORT_ERROR = None

# Reuse unified classification metadata convention from core
_CLASSIFICATION_META_KEY = core._CLASSIFICATION_META_KEY
_make_classification_meta_json = core._make_classification_meta_json
_load_classification_meta_from_user = core._load_classification_meta_from_user

_ds_get_rtdc_path = core._ds_get_rtdc_path
_find_true_runs = core._find_true_runs
_split_run = core._split_run
save_features_to_rtdc = core.save_features_to_rtdc

def classify_ds_ml(
    ds,
    mask: Optional[np.ndarray] = None,
    *,
    path_to_model: str,
    preprocessing_function: Callable[..., np.ndarray],
    extra_pixels: int = 15,
    overwrite: bool = True,  # kept for signature-compatibility; not used here
    show_progress: bool = True,
    model_ext: str = "_class",  # kept for signature-compatibility; not used here
    batch_size: int = 1024,
    num_workers: int = 6,
    prefetch_batches: int = 8,  # kept for signature-compatibility; not used here
    preprocess_chunk_size: int = 256,
    use_amp: Optional[bool] = None,
    preprocess_kwargs: Optional[dict] = None,
    cv_threads_per_worker: int = 2,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Classify events in an already-open RTDC dataset `ds` using the current ML pipeline,
    but only for events where `mask == True`.

    Returns:
        pred_class (np.uint8, len N),
        pred_prob  (np.float32, len N),
        meta       (dict) metadata describing the classifier and class names

    Notes:
      - Events outside `mask` are set to class=0 ("unclassified") and prob=0.
      - If mask is None, all events are classified.
    """
    preprocess_kwargs = preprocess_kwargs or {}

    n_events = len(ds)
    if mask is None:
        mask_arr = np.ones(n_events, dtype=bool)
    else:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape[0] != n_events:
            raise ValueError(f"mask length {mask_arr.shape[0]} != number of events {n_events}")

    # full-length outputs
    pred_class = np.zeros(n_events, dtype=np.uint8)
    pred_prob = np.zeros(n_events, dtype=np.float32)

    # Nothing to do
    if not np.any(mask_arr):
        meta = ds.config.as_dict(pop_filtering=True)
        meta.setdefault("user", {})
        _class_map = {0: "unclassified"}
        _classifier = {"name": "ML"}
        meta["user"][_CLASSIFICATION_META_KEY] = _make_classification_meta_json(
            class_feature="",
            class_map=_class_map,
            classifier=_classifier,
        )
        return pred_class, pred_prob, meta

    rtdc_path = _ds_get_rtdc_path(ds)

    model, classes, H0, W0 = load_model(path_to_model)
    device = _device()
    if use_amp is None:
        use_amp = (device == "cuda")

    # Metadata base
    meta = ds.config.as_dict(pop_filtering=True)
    meta.setdefault("user", {})

    # Human-readable classifier name for metadata
    try:
        classifier_name = str(Path(path_to_model).stem)
    except Exception:
        classifier_name = "ML"

    _class_map = {0: "unclassified"}
    _class_map.update({int(i)+1: str(n) for i, n in enumerate(list(classes))})
    _classifier = {
        "name": classifier_name,
        "cnn_backbone": getattr(model, "backbone_name", None),
    }
    meta["user"][_CLASSIFICATION_META_KEY] = _make_classification_meta_json(
        class_feature="",
        class_map=_class_map,
        classifier=_classifier,
    )

    # Prepare task list from mask runs
    runs = _find_true_runs(mask_arr)
    tasks: List[Tuple[int, int]] = []
    for s, e in runs:
        tasks.extend(_split_run(s, e, preprocess_chunk_size))

    # If no multiprocessing requested -> run in-process (still batched)
    if int(num_workers) <= 0:
        model.eval()

        idxs = []
        imgs = []
        scales = []

        def _flush():
            if not idxs:
                return
            x = np.stack(imgs, axis=0).astype(np.uint8, copy=False)
            x = torch.from_numpy(x).to(torch.float32)[:, None, :, :] / 255.0
            x = (x - 0.5) / 0.5
            x = x.to(device, non_blocking=True)

            sc = torch.tensor(scales, dtype=torch.float32, device=device)
            with torch.inference_mode(), _autocast_ctx(device, bool(use_amp)):
                out = model(x, sc)
                probs = F.softmax(out, dim=1)
                conf, cls = torch.max(probs, dim=1)

            cls_np = (cls.detach().cpu().numpy().astype(np.int32) + 1).astype(np.uint8)
            conf_np = conf.detach().cpu().numpy().astype(np.float32)
            pred_class[np.asarray(idxs, dtype=np.int64)] = cls_np
            pred_prob[np.asarray(idxs, dtype=np.int64)] = conf_np

            idxs.clear(); imgs.clear(); scales.clear()

        iterable = tasks
        if show_progress:
            iterable = tqdm(tasks, total=len(tasks), desc="Preprocess+infer", unit="chunk")

        for start, stop in iterable:
            for i in range(int(start), int(stop)):
                arr = preprocessing_function(ds, int(i), extra_pixels=extra_pixels, **preprocess_kwargs)
                if arr is None:
                    continue
                arr = np.asarray(arr)
                if arr.ndim != 2 or arr.dtype != np.uint8 or arr.shape[0] != H0:
                    continue
                Wvar = int(arr.shape[1])
                if Wvar <= 1:
                    continue
                scale_x = float(Wvar) / float(W0)
                if Wvar != W0:
                    arr = cv.resize(arr, (W0, H0), interpolation=cv.INTER_LINEAR)
                idxs.append(int(i))
                imgs.append(arr)
                scales.append(scale_x)
                if len(idxs) >= int(batch_size):
                    _flush()
        _flush()
        return pred_class, pred_prob, meta

    # Multiprocessing path (reuses existing worker protocol)
    in_q = mp.Queue(maxsize=max(2, int(prefetch_batches)))
    out_q = mp.Queue(maxsize=max(2, int(prefetch_batches)))

    procs = []
    for _ in range(int(num_workers)):
        p = mp.Process(
            target=_classify_worker,
            args=(rtdc_path, in_q, out_q, preprocessing_function, int(extra_pixels), dict(preprocess_kwargs), int(H0), int(W0), int(cv_threads_per_worker)),
            daemon=True,
        )
        p.start()
        procs.append(p)

    # Submit tasks
    task_iter = iter(tasks)
    inflight = 0
    max_inflight = max(1, int(prefetch_batches))

    def _submit_one():
        nonlocal inflight
        try:
            t = next(task_iter)
        except StopIteration:
            return False
        in_q.put(t)
        inflight += 1
        return True

    # Prime pipeline
    while inflight < max_inflight and _submit_one():
        pass

    buf_idxs: List[int] = []
    buf_imgs: List[np.ndarray] = []
    buf_scales: List[float] = []

    model.eval()

    def _infer_and_store(idxs_np: np.ndarray, imgs_np: np.ndarray, scales_np: np.ndarray):
        x = torch.from_numpy(imgs_np).to(torch.float32)[:, None, :, :] / 255.0
        x = (x - 0.5) / 0.5
        x = x.to(device, non_blocking=True)
        sc = torch.from_numpy(scales_np).to(device, non_blocking=True)
        with torch.inference_mode(), _autocast_ctx(device, bool(use_amp)):
            out = model(x, sc)
            probs = F.softmax(out, dim=1)
            conf, cls = torch.max(probs, dim=1)
        cls_np = (cls.detach().cpu().numpy().astype(np.int32) + 1).astype(np.uint8)
        conf_np = conf.detach().cpu().numpy().astype(np.float32)
        pred_class[idxs_np.astype(np.int64)] = cls_np
        pred_prob[idxs_np.astype(np.int64)] = conf_np

    pbar = None
    if show_progress:
        try:
            # progress in number of tasks processed (chunks)
            pbar = tqdm(total=len(tasks), desc="Preprocess+infer", unit="chunk")
        except Exception:
            pbar = None

    try:
        while inflight > 0:
            task_len, idxs_np, imgs_np, scales_np = out_q.get()
            inflight -= 1
            if pbar is not None:
                pbar.update(1)

            if idxs_np is not None:
                buf_idxs.extend(idxs_np.tolist())
                buf_imgs.extend(list(imgs_np))
                buf_scales.extend(scales_np.tolist())

            # keep pipeline full
            while inflight < max_inflight and _submit_one():
                pass

            while len(buf_idxs) >= int(batch_size):
                _infer_and_store(
                    np.asarray(buf_idxs[:batch_size], dtype=np.int32),
                    np.stack(buf_imgs[:batch_size], axis=0).astype(np.uint8, copy=False),
                    np.asarray(buf_scales[:batch_size], dtype=np.float32),
                )
                buf_idxs = buf_idxs[batch_size:]
                buf_imgs = buf_imgs[batch_size:]
                buf_scales = buf_scales[batch_size:]

        # flush remainder
        if buf_idxs:
            _infer_and_store(
                np.asarray(buf_idxs, dtype=np.int32),
                np.stack(buf_imgs, axis=0).astype(np.uint8, copy=False),
                np.asarray(buf_scales, dtype=np.float32),
            )
    finally:
        if pbar is not None:
            pbar.close()
        for _ in procs:
            in_q.put(None)
        for p in procs:
            p.join(timeout=5)
        for p in procs:
            if p.is_alive():
                p.terminate()

    return pred_class, pred_prob, meta


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

    # Work on subset indices
    idx_sel = np.flatnonzero(mask_arr)
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
        neut_tex_con = tex_con_avg[neutrophils]
        neut_max = float(np.max(neut_tex_con))
        neut_min = float(np.min(neut_tex_con))
        eosinophils = base6 & (tex_con_avg > neut_max)
        monocytes = base6 & (tex_con_avg < neut_min)

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





############################################################################
#
# common functions for preprocessing images of rtdc events before use in ML training and classification
#
############################################################################

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
    input and output images are considered to be grayscale uint8

    Minimal example preprocessing:
      - subtract background with get_img_diff (mean background ~128)
      - crop in x around the contour

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




############################################################################
#
# Here the machine learing part of the rtdc_toolbox begins
#
############################################################################

def _autocast_ctx(device: str, use_amp: bool):
    """Return an autocast context manager compatible with newer/older PyTorch."""
    if not use_amp or device != 'cuda':
        return contextlib.nullcontext()
    # New API (PyTorch >=2.0): torch.amp.autocast
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast(device_type='cuda', dtype=torch.float16)
    if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return contextlib.nullcontext()



def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _classify_worker(
    rtdc_path: str,
    in_q,
    out_q,
    preprocessing_function: Callable[..., np.ndarray],
    extra_pixels: int,
    preprocess_kwargs: dict,
    H0: int,
    W0: int,
    cv_threads: int = 2,
):
    """
    Worker for fast multicore classification:
    - Opens the RTDC dataset independently
    - Receives tasks as (start, stop) indices (stop exclusive)
    - Emits (task_len, idxs_np_int32, imgs_np_uint8, scales_np_float32)

    Notes:
    - Enforces uint8 grayscale output from preprocessing_function.
    - Resizes ONLY in x-direction to width W0 (height must already be H0).
    """
    try:
        # cv is imported at module scope as cv2
        cv.setUseOptimized(True)
        try:
            cv.setNumThreads(int(cv_threads))
        except Exception:
            pass
        _has_cv = True
    except Exception:
        _has_cv = False

    import dclab as _dclab

    with _dclab.new_dataset(rtdc_path) as ds:
        while True:
            task = in_q.get()
            if task is None:
                break
            start, stop = task
            task_len = int(stop - start)

            idxs = []
            imgs = []
            scales = []

            for i in range(int(start), int(stop)):
                arr = preprocessing_function(ds, int(i), extra_pixels=extra_pixels, **preprocess_kwargs)
                if arr is None:
                    continue
                arr = np.asarray(arr)

                if arr.ndim != 2:
                    continue
                if arr.dtype != np.uint8:
                    raise TypeError(f"preprocessing_function must return uint8, got {arr.dtype} (event {i})")

                if arr.shape[0] != H0:
                    raise ValueError(f"Preprocess returned H={arr.shape[0]} but model expects H0={H0} (event {i})")

                Wvar = int(arr.shape[1])
                if Wvar <= 1:
                    continue

                scale_x = float(Wvar) / float(W0)

                if Wvar != W0:
                    if _has_cv:
                        arr = cv.resize(arr, (W0, H0), interpolation=cv.INTER_LINEAR)
                    else:
                        arr = np.array(Image.fromarray(arr, mode="L").resize((W0, H0), resample=Image.BILINEAR))

                idxs.append(int(i))
                imgs.append(arr)
                scales.append(scale_x)

            if len(idxs) == 0:
                out_q.put((task_len, None, None, None))
            else:
                out_q.put((
                    task_len,
                    np.asarray(idxs, dtype=np.int32),
                    np.stack(imgs, axis=0).astype(np.uint8, copy=False),
                    np.asarray(scales, dtype=np.float32),
                ))
                

def _print_runtime_info():
    dev = _device()
    print(f"[rtdc_tools] torch={torch.__version__} | device={dev} | cuda_available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        try:
            print(f"[rtdc_tools] cuda_device={torch.cuda.get_device_name(0)} | capability={torch.cuda.get_device_capability(0)}", flush=True)
        except Exception:
            pass
        try:
            print(f"[rtdc_tools] cudnn={torch.backends.cudnn.version()} | cudnn_enabled={torch.backends.cudnn.enabled}", flush=True)
        except Exception:
            pass


# -----------------------------
# Imbalance config
# -----------------------------
@dataclass
class ImbalanceConfig:
    """
    Modular imbalance handling. Default is WeightedRandomSampler.
    Later you can swap strategy (e.g., class-weighted loss, focal loss, etc.).
    """
    strategy: str = "weighted_sampler"
    sampler_power: float = 1.0  # 1.0 -> inverse frequency; <1 softens


# -----------------------------
# Backbones (torchvision only)
# -----------------------------
def _build_backbone(name: str, pretrained: bool) -> Tuple[nn.Module, int, Optional[str]]:
    """
    Returns (feature_extractor, feature_dim, weights_name_for_metadata).

    Supported:
      - resnet18
      - resnet50
      - efficientnet_b0
      - convnext_tiny
    """
    name = str(name).lower().strip()

    if name == "resnet18":
        weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
        m = tvm.resnet18(weights=weights)
        feat_dim = int(m.fc.in_features)
        m.fc = nn.Identity()
        return m, feat_dim, str(weights) if weights is not None else None

    if name == "resnet50":
        weights = tvm.ResNet50_Weights.DEFAULT if pretrained else None
        m = tvm.resnet50(weights=weights)
        feat_dim = int(m.fc.in_features)
        m.fc = nn.Identity()
        return m, feat_dim, str(weights) if weights is not None else None

    if name == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = tvm.efficientnet_b0(weights=weights)
        feat_dim = int(m.classifier[-1].in_features)
        m.classifier[-1] = nn.Identity()
        return m, feat_dim, str(weights) if weights is not None else None

    if name == "convnext_tiny":
        weights = tvm.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        m = tvm.convnext_tiny(weights=weights)
        feat_dim = int(m.classifier[-1].in_features)
        m.classifier[-1] = nn.Identity()
        return m, feat_dim, str(weights) if weights is not None else None

    raise ValueError(f"Unsupported backbone '{name}'. Use: resnet18, resnet50, efficientnet_b0, convnext_tiny")


if _TORCH_AVAILABLE:
    # -----------------------------
    # Model: Backbone + scalar feature (scale_x)
    # -----------------------------
    class CNNWithScale(nn.Module):
        def __init__(self, backbone: str, num_classes: int, H0: int, W0: int, pretrained: bool = True):
            super().__init__()
            self.backbone_name = str(backbone)
            self.H0 = int(H0)
            self.W0 = int(W0)

            self.feature_extractor, feat_dim, self.weights_name = _build_backbone(backbone, pretrained=pretrained)
            self.classifier = nn.Linear(int(feat_dim) + 1, int(num_classes))

        def forward(self, x: torch.Tensor, scale_x: torch.Tensor) -> torch.Tensor:
            # x: (B,1,H,W) or (B,3,H,W)
            if x.dim() != 4:
                raise ValueError(f"x must be (B,C,H,W), got {tuple(x.shape)}")
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)

            x = F.interpolate(x, size=(self.H0, self.W0), mode="bilinear", align_corners=False)

            feat = self.feature_extractor(x)
            if feat.dim() != 2:
                feat = torch.flatten(feat, 1)

            if scale_x.dim() == 1:
                scale_x = scale_x[:, None]
            z = torch.cat([feat, scale_x], dim=1)
            return self.classifier(z)


    # -----------------------------
    # Data sources: ZIP or folder
    # -----------------------------
    def _list_zip_images(zip_path: str, img_ext: str) -> List[str]:
        img_ext = img_ext.lower()
        with zipfile.ZipFile(zip_path, "r") as zf:
            return [n for n in zf.namelist() if n.lower().endswith(img_ext) and not n.endswith("/")]


    def _infer_classes_from_zip(zip_path: str, img_ext: str) -> List[str]:
        names = _list_zip_images(zip_path, img_ext)
        classes = sorted({n.split("/")[0] for n in names if "/" in n})
        if not classes:
            raise ValueError("No class folders found in ZIP. Expected ZIP/<class>/<image>.")
        return classes


    def _infer_classes_from_folder(folder: str) -> List[str]:
        classes = []
        for name in sorted(os.listdir(folder)):
            p = os.path.join(folder, name)
            if os.path.isdir(p):
                classes.append(name)
        if not classes:
            raise ValueError("No class subfolders found. Expected folder/<class>/<image>.")
        return classes


    class PreprocessedDataset(Dataset):
        """
        Unified dataset for preprocessed images from either:
          - zip file (zip=True)
          - folder with class subfolders (zip=False)

        Optional RAM cache:
          - cache_in_ram=True loads all images into RAM once.
          - Recommended: cache_in_ram=True with num_workers=0 (no duplication).
        """

        def __init__(
            self,
            path_to_training_set: str,
            classes: Sequence[str],
            zip: bool = True,
            img_ext: str = ".png",
            W0: int = 160,
            train: bool = True,
            brightness_jitter: float = 0.0,
            contrast_jitter: float = 0.0,
            normalize: bool = True,
            cache_in_ram: bool = False,
        ):
            self.path = str(path_to_training_set)
            self.zip = bool(zip)
            self.img_ext = str(img_ext)
            self.W0 = int(W0)
            self.train = bool(train)
            self.brightness_jitter = float(brightness_jitter)
            self.contrast_jitter = float(contrast_jitter)
            self.normalize = bool(normalize)
            self.cache_in_ram = bool(cache_in_ram)

            self.classes = list(classes)
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

            self.samples: List[Tuple[str, int]] = []

            if self.zip:
                with zipfile.ZipFile(self.path, "r") as zf:
                    for n in zf.namelist():
                        if n.endswith("/") or (not n.lower().endswith(self.img_ext.lower())):
                            continue
                        cls = n.split("/")[0]
                        if cls in self.class_to_idx:
                            self.samples.append((n, self.class_to_idx[cls]))
            else:
                for cls in self.classes:
                    cls_dir = os.path.join(self.path, cls)
                    if not os.path.isdir(cls_dir):
                        continue
                    for fn in os.listdir(cls_dir):
                        if fn.lower().endswith(self.img_ext.lower()):
                            self.samples.append((os.path.join(cls_dir, fn), self.class_to_idx[cls]))

            if not self.samples:
                raise ValueError("No images found matching img_ext and class folders.")

            self.H0 = self._peek_height()

            self._cache_imgs: Optional[List[np.ndarray]] = None
            self._cache_labels: Optional[List[int]] = None
            if self.cache_in_ram:
                self._build_cache()

        def _peek_height(self) -> int:
            if self.zip:
                name, _ = self.samples[0]
                with zipfile.ZipFile(self.path, "r") as zf:
                    with zf.open(name) as f:
                        arr = np.asarray(Image.open(f).convert("L")).copy()
                return int(arr.shape[0])
            fp, _ = self.samples[0]
            arr = np.asarray(Image.open(fp).convert("L")).copy()
            return int(arr.shape[0])

        def _build_cache(self):
            print(f"[rtdc_tools] Caching dataset in RAM ({len(self.samples)} images)...", flush=True)
            imgs: List[np.ndarray] = []
            labels: List[int] = []
            if self.zip:
                with zipfile.ZipFile(self.path, "r") as zf:
                    for name, y in tqdm(self.samples, desc="Cache ZIP", leave=True):
                        with zf.open(name) as f:
                            arr = np.asarray(Image.open(f).convert("L")).copy()
                        imgs.append(arr)
                        labels.append(int(y))
            else:
                for fp, y in tqdm(self.samples, desc="Cache folder", leave=True):
                    arr = np.asarray(Image.open(fp).convert("L")).copy()
                    imgs.append(arr)
                    labels.append(int(y))
            self._cache_imgs = imgs
            self._cache_labels = labels
            print("[rtdc_tools] Cache done.", flush=True)

        def __len__(self) -> int:
            return len(self.samples)

        def _load_array(self, idx: int) -> Tuple[np.ndarray, int, str]:
            if self.cache_in_ram:
                return self._cache_imgs[idx], int(self._cache_labels[idx]), self.samples[idx][0]

            name, y = self.samples[idx]
            if self.zip:
                with zipfile.ZipFile(self.path, "r") as zf:
                    with zf.open(name) as f:
                        arr = np.asarray(Image.open(f).convert("L")).copy()
            else:
                arr = np.asarray(Image.open(name).convert("L")).copy()
            return arr, int(y), name

        def __getitem__(self, idx: int):
            arr, y, name = self._load_array(idx)

            if arr.shape[0] != self.H0:
                raise ValueError(f"Image height mismatch: expected {self.H0}, got {arr.shape[0]} for {name}")

            Wvar = int(arr.shape[1])
            scale_x = float(Wvar) / float(self.W0)

            x = torch.from_numpy(arr).to(torch.float32)[None, :, :]
            x = x / 255.0

            if self.train and (self.brightness_jitter > 0 or self.contrast_jitter > 0):
                if self.brightness_jitter > 0:
                    delta = float(torch.empty(1).uniform_(-self.brightness_jitter, self.brightness_jitter).item())
                    x = x + delta
                if self.contrast_jitter > 0:
                    factor = float(torch.empty(1).uniform_(1.0 - self.contrast_jitter, 1.0 + self.contrast_jitter).item())
                    x = (x - 0.5) * factor + 0.5
                x = torch.clamp(x, 0.0, 1.0)

            if self.normalize:
                x = (x - 0.5) / 0.5

            x = F.interpolate(x[None, :, :, :], size=(self.H0, self.W0), mode="bilinear", align_corners=False)[0]

            return x, torch.tensor(scale_x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


    def _make_sampler(labels: Sequence[int], cfg: ImbalanceConfig) -> WeightedRandomSampler:
        counts = Counter(labels)
        weights = [1.0 / (counts[int(y)] ** cfg.sampler_power) for y in labels]
        return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


    def _eval_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> Dict[str, Any]:
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        y_true: List[int] = []
        y_pred: List[int] = []

        for x, scale_x, y in tqdm(loader, desc="Eval", leave=False):
            x = x.to(device, non_blocking=True)
            scale_x = scale_x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x, scale_x)
            loss = criterion(logits, y)

            total_loss += float(loss.item()) * int(y.shape[0])
            pred = torch.argmax(logits, dim=1)

            correct += int((pred == y).sum().item())
            total += int(y.shape[0])

            y_true.extend(y.detach().cpu().tolist())
            y_pred.extend(pred.detach().cpu().tolist())

        return {"loss": total_loss / max(1, total), "acc": correct / max(1, total), "y_true": y_true, "y_pred": y_pred}


    def train_model(
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
        imbalance_cfg: Optional[ImbalanceConfig] = None,
        seed: int = 123,
        num_workers: int = 0,
        pretrained: bool = True,
        cache_in_ram: bool = False,
    ) -> str:
        _print_runtime_info()
        os.makedirs(training_results_dir, exist_ok=True)
        imbalance_cfg = imbalance_cfg or ImbalanceConfig()

        classes = _infer_classes_from_zip(path_to_training_set, img_ext) if zip else _infer_classes_from_folder(path_to_training_set)

        full_ds = PreprocessedDataset(
            path_to_training_set=path_to_training_set,
            classes=classes,
            zip=zip,
            img_ext=img_ext,
            W0=W0,
            train=True,
            brightness_jitter=brightness_jitter,
            contrast_jitter=contrast_jitter,
            normalize=True,
            cache_in_ram=cache_in_ram,
        )
        H0 = full_ds.H0
        src = "ZIP" if zip else "folder"
        print(f"[rtdc_tools] source={src} | samples={len(full_ds)} | classes={classes} | H0={H0} | W0={W0}", flush=True)
        if cache_in_ram:
            print("[rtdc_tools] cache_in_ram=True (recommend num_workers=0 to avoid duplication across workers).", flush=True)

        g = torch.Generator().manual_seed(int(seed))
        n_total = len(full_ds)
        n_val = int(round(float(val_split) * n_total))
        n_train = n_total - n_val
        tr_split, va_split = torch.utils.data.random_split(full_ds, [n_train, n_val], generator=g)

        val_ds_full = PreprocessedDataset(
            path_to_training_set=path_to_training_set,
            classes=classes,
            zip=zip,
            img_ext=img_ext,
            W0=W0,
            train=False,
            brightness_jitter=0.0,
            contrast_jitter=0.0,
            normalize=True,
            cache_in_ram=cache_in_ram,
        )

        train_ds = torch.utils.data.Subset(full_ds, tr_split.indices)
        val_ds = torch.utils.data.Subset(val_ds_full, va_split.indices)

        train_labels = [full_ds.samples[i][1] for i in train_ds.indices]
        sampler = _make_sampler(train_labels, imbalance_cfg) if imbalance_cfg.strategy == "weighted_sampler" else None

        def _pf():
            return 4 if int(num_workers) > 0 else None

        train_loader = DataLoader(
            train_ds,
            batch_size=int(batch_size),
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=int(num_workers),
            pin_memory=True,
            persistent_workers=(int(num_workers) > 0),
            prefetch_factor=_pf(),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=True,
            persistent_workers=(int(num_workers) > 0),
            prefetch_factor=_pf(),
        )

        device = _device()
        model = CNNWithScale(backbone=cnn_backbone, num_classes=len(classes), H0=H0, W0=W0, pretrained=pretrained).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

        best = {"acc": -1.0, "epoch": -1, "state": None}
        history: List[Dict[str, float]] = []

        for epoch in range(1, int(n_epochs) + 1):
            model.train()
            t0 = time.time()
            run_loss = 0.0
            correct = 0
            total = 0

            for x, scale_x, y in tqdm(train_loader, desc=f"Train epoch {epoch}/{n_epochs}", leave=True):
                x = x.to(device, non_blocking=True)
                scale_x = scale_x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(x, scale_x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                run_loss += float(loss.item()) * int(y.shape[0])
                pred = torch.argmax(logits, dim=1)
                correct += int((pred == y).sum().item())
                total += int(y.shape[0])

            train_loss = run_loss / max(1, total)
            train_acc = correct / max(1, total)

            val_metrics = _eval_epoch(model, val_loader, criterion, device=device)
            dt = time.time() - t0

            history.append({
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_metrics["loss"]),
                "val_acc": float(val_metrics["acc"]),
                "seconds": float(dt),
            })

            print(
                f"Epoch {epoch:03d}/{n_epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f} | "
                f"{dt:.1f}s",
                flush=True
            )


            # --- per-epoch artifacts (so you can pick an earlier epoch if needed)
            epoch_dir = os.path.join(training_results_dir, f"epoch_{epoch:03d}")
            os.makedirs(epoch_dir, exist_ok=True)

            # Confusion matrix for this epoch (val)
            cm_epoch = confusion_matrix(val_metrics["y_true"], val_metrics["y_pred"], labels=list(range(len(classes))))
            _save_confusion_matrix_percent(
                cm_epoch,
                classes,
                out_path=os.path.join(epoch_dir, f"{model_name}_confusion.png"),
                title=f"Confusion matrix (val) - epoch {epoch}"
            )

            # Save checkpoint bundle (self-contained)
            ckpt_bundle = {
                "format": "rtdc_tools",
                "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
                "backbone": str(cnn_backbone),
                "backbone_weights": getattr(model, "weights_name", None),
                "classes": list(classes),
                "H0": int(H0),
                "W0": int(W0),
                "normalize": {"mode": "x=(x/255-0.5)/0.5"},
                "imbalance": {"strategy": imbalance_cfg.strategy, "sampler_power": float(imbalance_cfg.sampler_power)},
                "epoch": int(epoch),
                "model_state_dict": model.state_dict(),
            }
            torch.save(ckpt_bundle, os.path.join(epoch_dir, "model.pt"))

            # Save per-epoch JSON (human-readable; not required for inference)
            ckpt_meta = {
                "model_name": model_name,
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_metrics["loss"]),
                "val_acc": float(val_metrics["acc"]),
                "seconds": float(dt),
                "classes": list(classes),
                "backbone": str(cnn_backbone),
                "backbone_weights": getattr(model, "weights_name", None),
                "H0": int(H0),
                "W0": int(W0),
                "img_ext": img_ext,
                "source_is_zip": bool(zip),
                "cache_in_ram": bool(cache_in_ram),
                "brightness_jitter": float(brightness_jitter),
                "contrast_jitter": float(contrast_jitter),
                "imbalance": {"strategy": imbalance_cfg.strategy, "sampler_power": float(imbalance_cfg.sampler_power)},
            }
            with open(os.path.join(epoch_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(ckpt_meta, f, indent=2)

            # Update history TSV each epoch (handy for monitoring / later comparisons)
            _write_history_tsv(history, os.path.join(training_results_dir, f"{model_name}_history.tsv"))

            if float(val_metrics["acc"]) > float(best["acc"]):
                best["acc"] = float(val_metrics["acc"])
                best["epoch"] = int(epoch)
                best["state"] = copy.deepcopy(model.state_dict())

        if best["state"] is not None:
            model.load_state_dict(best["state"])

        final_val = _eval_epoch(model, val_loader, criterion, device=device)
        cm = confusion_matrix(final_val["y_true"], final_val["y_pred"], labels=list(range(len(classes))))

        # Root-level artifacts
        _save_accuracy_plot(history, os.path.join(training_results_dir, f"{model_name}_acc.png"))
        _write_history_tsv(history, os.path.join(training_results_dir, f"{model_name}_history.tsv"))
        _save_confusion_matrix_percent(cm, classes, os.path.join(training_results_dir, f"{model_name}_confusion.png"))
    
        model_filename = f"{model_name}_" + "_".join(classes) + ".pt"
        model_path = os.path.join(training_results_dir, model_filename)

        bundle = {
            "format": "rtdc_tools",
            "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "backbone": str(cnn_backbone),
            "backbone_weights": getattr(model, "weights_name", None),
            "classes": list(classes),
            "H0": int(H0),
            "W0": int(W0),
            "normalize": {"mode": "x=(x/255-0.5)/0.5"},
            "imbalance": {"strategy": imbalance_cfg.strategy, "sampler_power": float(imbalance_cfg.sampler_power)},
            "model_state_dict": model.state_dict(),
        }
        torch.save(bundle, model_path)

        meta = {
            "model_path": model_path,
            "model_name": model_name,
            "classes": list(classes),
            "backbone": str(cnn_backbone),
            "backbone_weights": getattr(model, "weights_name", None),
            "H0": int(H0),
            "W0": int(W0),
            "img_ext": img_ext,
            "source_is_zip": bool(zip),
            "cache_in_ram": bool(cache_in_ram),
            "n_epochs": int(n_epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "brightness_jitter": float(brightness_jitter),
            "contrast_jitter": float(contrast_jitter),
            "val_split": float(val_split),
            "seed": int(seed),
            "best_val_acc": float(best["acc"]),
            "best_epoch": int(best["epoch"]),
            "history": history,
            "val_confusion_matrix": cm.tolist(),
            "class_counts_train": dict(Counter(train_labels)),
            "preprocessing_function_note": (
                "Training images are assumed already preprocessed with the same preprocessing function used for RTDC inference. "
                "This function is not invoked during training."
            ),
        }
        with open(os.path.join(training_results_dir, f"{model_name}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return model_path



    def _write_history_tsv(history: List[Dict[str, float]], out_path: str) -> None:
        # Write a simple TSV for later comparison across models
        cols = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"]
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\t".join(cols) + "\n")
            for h in history:
                f.write("\t".join(str(h.get(c, "")) for c in cols) + "\n")


    def _save_confusion_matrix_percent(cm: np.ndarray, classes: List[str], out_path: str, title: str = "Confusion matrix (val)") -> None:
        """
        Save confusion matrix image with per-cell percentages (row-normalized).
        No colorbar (legend) to keep it clean.
        """
        cm = np.asarray(cm, dtype=np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        pct = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0) * 100.0

        plt.figure(figsize=(max(6, 0.6 * len(classes)), max(5, 0.5 * len(classes))))
        plt.imshow(pct, interpolation="nearest")
        plt.title(title)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, ha="right")
        plt.yticks(tick_marks, classes)
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

        thresh = (pct.max() if pct.size else 0) * 0.5
        for i in range(pct.shape[0]):
            for j in range(pct.shape[1]):
                val = pct[i, j]
                txt = f"{val:.1f}%" if row_sums[i, 0] > 0 else "-"
                plt.text(j, i, txt, ha="center", va="center", color="white" if val > thresh else "black", fontsize=9)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()


    def _save_accuracy_plot(history: List[Dict[str, float]], out_path: str) -> None:
        epochs = [h["epoch"] for h in history]
        train_acc = [h["train_acc"] for h in history]
        val_acc = [h["val_acc"] for h in history]

        plt.figure()
        plt.plot(epochs, train_acc, label="train_acc")
        plt.plot(epochs, val_acc, label="val_acc")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()



    def load_model(model_path: str) -> Tuple[nn.Module, List[str], int, int]:
        bundle = torch.load(model_path, map_location=_device())
        classes = list(bundle["classes"])
        H0 = int(bundle["H0"])
        W0 = int(bundle["W0"])
        backbone = str(bundle["backbone"])

        model = CNNWithScale(backbone=backbone, num_classes=len(classes), H0=H0, W0=W0, pretrained=False).to(_device())
        model.load_state_dict(bundle["model_state_dict"])
        model.eval()
        return model, classes, H0, W0



    def classify_rtdc_ml(
        path_to_rtdc: str,
        path_to_model: str,
        preprocessing_function: Callable[..., np.ndarray],
        class_feature: str = "userdef1",
        prob_feature: str = "userdef2",
        extra_pixels: int = 15,
        overwrite: bool = True,
        show_progress: bool = True,
        model_ext: str = "_class",
        batch_size: int = 1024,
        num_workers: int = 6,
        prefetch_batches: int = 8,
        preprocess_chunk_size: int = 256,
        use_amp: Optional[bool] = None,
        preprocess_kwargs: Optional[dict] = None,
        cv_threads_per_worker: int = 2
    ) -> str:
        """
        Fast RTDC classification for very large datasets (e.g. ~1e6 events).

        This function keeps its original API and behavior, but now delegates
        classification to `classify_ds_ml` and saving/linking to `save_features_to_rtdc`.
        """
        # output path convention: append model_ext before ".rtdc"
        if str(path_to_rtdc).lower().endswith(".rtdc"):
            out_path = str(path_to_rtdc)[:-5] + str(model_ext) + ".rtdc"
        else:
            out_path = str(path_to_rtdc) + str(model_ext) + ".rtdc"

        if os.path.exists(out_path) and not overwrite:
            print(f"[SKIP] file aready exists (you can set overwrite to True): {out_path}", flush=True)
            return out_path

        with dclab.new_dataset(path_to_rtdc) as ds:
            pred_class, pred_prob, meta = classify_ds_ml(
                ds,
                mask=None,
                path_to_model=path_to_model,
                preprocessing_function=preprocessing_function,
                extra_pixels=extra_pixels,
                overwrite=overwrite,
                show_progress=show_progress,
                model_ext=model_ext,
                batch_size=batch_size,
                num_workers=num_workers,
                prefetch_batches=prefetch_batches,
                preprocess_chunk_size=preprocess_chunk_size,
                use_amp=use_amp,
                preprocess_kwargs=preprocess_kwargs,
                cv_threads_per_worker=cv_threads_per_worker,
            )

        # Store the classification result in a new RTDC file linked to the input dataset.
        save_features_to_rtdc(
            basin_rtdc_path=str(path_to_rtdc),
            output_rtdc_path=str(out_path),
            feature_dict={str(class_feature): pred_class, str(prob_feature): pred_prob},
            metadata=meta,
            overwrite=True,
        )

        return out_path

else:
    # ML functionality placeholders (torch/torchvision unavailable)
    pass