
"""PyTorch image-model training and inference for RT-DC data.

rtdc-toolbox framework version 2.0 (August 2026)

Main public functions
---------------------
    train_model_on_images
        Train a CNN from preprocessed grayscale images stored in a ZIP archive
        or a folder hierarchy.
    load_image_model
        Reconstruct a saved image model and load its weights.
    ImageImbalanceConfig
        Configure weighted sampling for imbalanced image classes.
    classify_ds_images_ml
        Run image-only inference on an already-open RTDC dataset and return
        class labels, probabilities, and classifier information.
    classify_rtdc_images_ml
        Run image-only inference, allocate output ``userdefN`` features, write
        a basin-linked RTDC file, and register metadata format 2.0.

The module intentionally uses ``image`` in all public image-model names so that
future feature-only or multimodal models can coexist without ambiguity.
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
import queue
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
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import dclab

# Import core utilities (torch-free)
import rtdc_toolbox as core

# Torch/torchvision are REQUIRED for this module
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as tvm

FRAMEWORK_VERSION = "2.0 (August 2026)"

__all__ = [
    "FRAMEWORK_VERSION",
    "ImageImbalanceConfig",
    "ImageCNNWithScale",
    "PreprocessedImageDataset",
    "train_model_on_images",
    "load_image_model",
    "classify_ds_images_ml",
    "classify_rtdc_images_ml",
]

_TORCH_AVAILABLE = True
_TORCH_IMPORT_ERROR = None

# Reuse unified classification metadata convention from core




CLASSIFICATION_METADATA_VERSION = core.CLASSIFICATION_METADATA_VERSION
_register_classifier_metadata = core._register_classifier_metadata
_ds_get_rtdc_path = core._ds_get_rtdc_path
_find_true_runs = core._find_true_runs
_split_run = core._split_run
save_features_to_rtdc = core.save_features_to_rtdc


def _prepare_inference_image(
    arr: np.ndarray,
    H0: int,
    W0: int,
    *,
    use_cv: bool = True,
) -> Tuple[np.ndarray, float]:
    """Prepare one grayscale uint8 image for ML inference.

    Processing order:
      1. If the input height differs from H0, resize to H0 while preserving
         the original aspect ratio.
      2. Calculate scale_x from the aspect-ratio-preserved width.
      3. Resize the width to W0, producing the final H0 x W0 model input.

    Returns
    -------
    image : np.ndarray
        Grayscale uint8 image with shape (H0, W0).
    scale_x : float
        Width after height normalization divided by W0.
    """
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        raise TypeError(f"Expected a uint8 image, got {arr.dtype}")

    Hvar, Wvar = map(int, arr.shape)
    if Hvar <= 1 or Wvar <= 1:
        raise ValueError(f"Image is too small for inference: shape={arr.shape}")

    # First normalize the height while retaining the original aspect ratio.
    if Hvar != int(H0):
        width_at_H0 = max(2, int(round(Wvar * float(H0) / float(Hvar))))
        if use_cv:
            interpolation = cv.INTER_AREA if int(H0) < Hvar else cv.INTER_LINEAR
            arr = cv.resize(arr, (width_at_H0, int(H0)), interpolation=interpolation)
        else:
            arr = np.array(
                Image.fromarray(arr, mode="L").resize(
                    (width_at_H0, int(H0)),
                    resample=Image.BILINEAR,
                )
            )

    # Preserve the width information that remains after height normalization.
    width_at_H0 = int(arr.shape[1])
    scale_x = float(width_at_H0) / float(W0)

    # Then normalize the width for the fixed-size CNN input.
    if width_at_H0 != int(W0):
        if use_cv:
            interpolation = cv.INTER_AREA if int(W0) < width_at_H0 else cv.INTER_LINEAR
            arr = cv.resize(arr, (int(W0), int(H0)), interpolation=interpolation)
        else:
            arr = np.array(
                Image.fromarray(arr, mode="L").resize(
                    (int(W0), int(H0)),
                    resample=Image.BILINEAR,
                )
            )

    return arr.astype(np.uint8, copy=False), scale_x






def _find_x_overlapping_events(
    ds: Any,
    *,
    candidate_mask: Optional[np.ndarray] = None,
    min_overlap: float = 0.0,
    min_overlap_fraction: float = 0.0,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mark same-frame events whose scalar x intervals overlap.

    The check uses only the scalar RTDC features ``frame``, ``pos_x``, and
    ``size_x``. In dclab, ``pos_x`` and ``size_x`` are expressed in micrometres
    (µm), so the interval arithmetic and ``min_overlap`` are also in µm. It
    never reads event contours.

    Each event is represented by the horizontal interval
    ``[pos_x - size_x/2, pos_x + size_x/2]``.  Both members of a qualifying
    same-frame pair are marked as ambiguous.

    Parameters
    ----------
    ds:
        Open RTDC dataset containing valid ``frame``, ``pos_x``, and ``size_x``
        features.
    candidate_mask:
        Optional Boolean mask.  Only frames containing at least one requested
        event are checked, but all events in those frames can act as neighbours.
    min_overlap:
        Minimum overlap width in µm, matching the dclab units of ``pos_x`` and
        ``size_x``. The default 0 marks every strictly positive overlap.
    min_overlap_fraction:
        Minimum overlap width divided by the narrower event width.

    Returns
    -------
    ambiguous:
        Boolean array marking events that overlap another event in x within the
        same frame.
    affected_frames:
        Frame identifiers containing at least one qualifying overlap pair.
    """
    required = ("frame", "pos_x", "size_x")
    missing = [feature for feature in required if feature not in ds.features]
    if missing:
        raise KeyError(
            "X-overlap exclusion requires RTDC features: " + ", ".join(missing)
        )
    if min_overlap < 0:
        raise ValueError("min_overlap must be >= 0")
    if not 0 <= min_overlap_fraction <= 1:
        raise ValueError("min_overlap_fraction must be between 0 and 1")

    n_events = len(ds)
    ambiguous = np.zeros(n_events, dtype=bool)
    if n_events == 0:
        return ambiguous, np.asarray([], dtype=np.int64)

    if candidate_mask is None:
        candidate = np.ones(n_events, dtype=bool)
    else:
        candidate = np.asarray(candidate_mask, dtype=bool)
        if candidate.ndim != 1 or candidate.shape[0] != n_events:
            raise ValueError(
                f"candidate_mask must have shape ({n_events},); got {candidate.shape}"
            )

    # Load the three fixed-size scalar features once.  This is much faster than
    # accessing variable-length contour datasets event by event.
    frames = np.asarray(ds["frame"][:])
    pos_x = np.asarray(ds["pos_x"][:], dtype=np.float64)
    size_x = np.asarray(ds["size_x"][:], dtype=np.float64)
    for name, values in (("frame", frames), ("pos_x", pos_x), ("size_x", size_x)):
        if values.ndim != 1 or values.shape[0] != n_events:
            raise ValueError(
                f"RTDC feature '{name}' must have shape ({n_events},); "
                f"got {values.shape}"
            )

    # RTDC events are normally ordered by frame.  Avoid sorting in that common
    # case; otherwise perform one stable sort to group equal frame identifiers.
    if n_events < 2 or np.all(frames[1:] >= frames[:-1]):
        order = np.arange(n_events, dtype=np.int64)
        grouped_frames = frames
    else:
        order = np.argsort(frames, kind="stable")
        grouped_frames = frames[order]

    boundaries = np.concatenate((
        np.asarray([0], dtype=np.int64),
        np.flatnonzero(grouped_frames[1:] != grouped_frames[:-1]).astype(np.int64) + 1,
        np.asarray([n_events], dtype=np.int64),
    ))
    starts = boundaries[:-1]
    stops = boundaries[1:]
    sizes = stops - starts

    candidate_counts = np.add.reduceat(
        candidate[order].astype(np.int8, copy=False),
        starts,
    )
    selected_groups = np.flatnonzero((sizes >= 2) & (candidate_counts > 0))
    if selected_groups.size == 0:
        if verbose:
            print(
                "[2/4] No requested events occur in frames containing multiple "
                "events; x-overlap exclusion is complete.",
                flush=True,
            )
        return ambiguous, np.asarray([], dtype=frames.dtype)

    if verbose:
        print(
            f"[2/4] Checking scalar x-overlap in {selected_groups.size:,} "
            "multi-event frames using frame, pos_x, and size_x...",
            flush=True,
        )

    affected_frame_values: List[Any] = []

    # Vectorize the very common case of exactly two events in a frame.
    pair_groups = selected_groups[sizes[selected_groups] == 2]
    if pair_groups.size:
        first = order[starts[pair_groups]]
        second = order[starts[pair_groups] + 1]

        valid = (
            np.isfinite(pos_x[first])
            & np.isfinite(pos_x[second])
            & np.isfinite(size_x[first])
            & np.isfinite(size_x[second])
            & (size_x[first] > 0)
            & (size_x[second] > 0)
        )

        left_first = pos_x[first] - 0.5 * size_x[first]
        right_first = pos_x[first] + 0.5 * size_x[first]
        left_second = pos_x[second] - 0.5 * size_x[second]
        right_second = pos_x[second] + 0.5 * size_x[second]

        overlap = np.minimum(right_first, right_second) - np.maximum(
            left_first, left_second
        )
        narrower = np.minimum(size_x[first], size_x[second])
        fraction = np.divide(
            overlap,
            narrower,
            out=np.zeros_like(overlap),
            where=narrower > 0,
        )
        qualifies = (
            valid
            & (overlap > 0)
            & (overlap >= min_overlap)
            & (fraction >= float(min_overlap_fraction))
        )

        if np.any(qualifies):
            ambiguous[first[qualifies]] = True
            ambiguous[second[qualifies]] = True
            affected_frame_values.extend(
                grouped_frames[starts[pair_groups[qualifies]]].tolist()
            )

    # Frames with three or more events are uncommon.  Check their small set of
    # intervals with a simple left-to-right sweep.
    larger_groups = selected_groups[sizes[selected_groups] > 2]
    for group_index in larger_groups:
        lo = int(starts[group_index])
        hi = int(stops[group_index])
        indices = np.asarray(order[lo:hi], dtype=np.int64)

        valid = (
            np.isfinite(pos_x[indices])
            & np.isfinite(size_x[indices])
            & (size_x[indices] > 0)
        )
        indices = indices[valid]
        if indices.size < 2:
            continue

        left = pos_x[indices] - 0.5 * size_x[indices]
        right = pos_x[indices] + 0.5 * size_x[indices]
        local_order = np.argsort(left, kind="stable")
        frame_affected = False

        for position, local_i in enumerate(local_order[:-1]):
            idx_i = int(indices[local_i])
            right_i = float(right[local_i])
            width_i = float(size_x[idx_i])

            for local_j in local_order[position + 1:]:
                if float(left[local_j]) >= right_i:
                    break
                idx_j = int(indices[local_j])
                overlap = min(right_i, float(right[local_j])) - max(
                    float(left[local_i]), float(left[local_j])
                )
                if overlap <= 0:
                    continue
                narrower = min(width_i, float(size_x[idx_j]))
                fraction = overlap / narrower if narrower > 0 else 0.0
                if (
                    overlap >= min_overlap
                    and fraction >= float(min_overlap_fraction)
                ):
                    ambiguous[idx_i] = True
                    ambiguous[idx_j] = True
                    frame_affected = True

        if frame_affected:
            frame_value = grouped_frames[lo]
            affected_frame_values.append(
                frame_value.item() if hasattr(frame_value, "item") else frame_value
            )

    affected_frames = np.asarray(affected_frame_values, dtype=frames.dtype)
    if verbose:
        print(
            f"[2/4] X-overlap check complete: {int(ambiguous.sum()):,} ambiguous "
            f"events in {affected_frames.size:,} frames.",
            flush=True,
        )
    return ambiguous, affected_frames



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
    """Classify selected events in an open dataset using an image model.

    Events outside ``mask``, events excluded because of same-frame x-overlap,
    and events that fail preprocessing remain class 0 with probability 0. The
    function does not allocate RTDC features, register metadata, or write files.

    Returns
    -------
    classes, probabilities, result
        Full-length class and probability arrays plus a dictionary containing
        the class map and classifier metadata.
    """
    preprocessing_function = preprocessing_function or core.preprocess_rtdc_image
    preprocess_kwargs = dict(preprocess_kwargs or {})

    batch_size = int(batch_size)
    num_workers = int(num_workers)
    prefetch_batches = int(prefetch_batches)
    preprocess_chunk_size = int(preprocess_chunk_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if prefetch_batches <= 0:
        raise ValueError("prefetch_batches must be > 0")
    if preprocess_chunk_size <= 0:
        raise ValueError("preprocess_chunk_size must be > 0")

    n_events = len(ds)
    requested_mask = core._normalize_event_mask(mask, n_events)
    selection_info = core._make_selection_info(
        mask,
        requested_mask,
        description=mask_description,
    )

    model_path = Path(path_to_model)
    default_name = model_path.stem or "image_classifier"
    model_file = model_path.name or str(path_to_model)
    resolved_name = str(classifier_name or default_name).strip()
    if not resolved_name:
        raise ValueError("classifier_name must not be empty")

    if verbose:
        print(f"[1/4] Loading image model: {model_file}", flush=True)
    model, model_classes, model_height, model_width = load_image_model(path_to_model)
    device = _device()
    if verbose:
        print(
            f"[1/4] Image model loaded on {device}: {len(model_classes)} classes, "
            f"input H0={model_height}, W0={model_width}.",
            flush=True,
        )
    if use_amp is None:
        use_amp = device == "cuda"

    classes = np.zeros(n_events, dtype=np.uint8)
    probabilities = np.zeros(n_events, dtype=np.float32)

    # Exclude ambiguous same-frame events before preprocessing and inference.
    ambiguous_mask = np.zeros(n_events, dtype=bool)
    affected_frames = np.asarray([], dtype=int)
    if exclude_x_overlapping_events and np.any(requested_mask):
        ambiguous_mask, affected_frames = _find_x_overlapping_events(
            ds,
            candidate_mask=requested_mask,
            min_overlap=float(min_x_overlap),
            min_overlap_fraction=float(min_x_overlap_fraction),
            verbose=verbose,
        )
    elif verbose and not exclude_x_overlapping_events:
        print("[2/4] Same-frame x-overlap exclusion is disabled.", flush=True)

    excluded_overlap_mask = requested_mask & ambiguous_mask
    inference_mask = requested_mask & ~ambiguous_mask
    n_overlap_excluded = int(excluded_overlap_mask.sum())
    if verbose and n_overlap_excluded:
        print(
            f"WARNING: Excluding {n_overlap_excluded:,} requested events because "
            "their horizontal extent overlaps another event in the same frame. "
            "They remain class 0 with probability 0.",
            flush=True,
        )

    if verbose:
        print(
            f"Images with a height different from model H0={model_height} are "
            "rescaled with preserved aspect ratio before the width is rescaled "
            f"to W0={model_width}.",
            flush=True,
        )

    class_map = {0: "unclassified"}
    class_map.update(
        {index + 1: str(name) for index, name in enumerate(model_classes)}
    )
    overlap_info = {
        "mode": "exclude" if exclude_x_overlapping_events else "ignore",
        "min_x_overlap": float(min_x_overlap),
        "min_x_overlap_fraction": float(min_x_overlap_fraction),
        "check_method": "scalar_pos_x_size_x",
        "n_overlapping_events_total": int(ambiguous_mask.sum()),
        "n_excluded_events_from_requested_mask": n_overlap_excluded,
        "n_affected_frames_total": int(len(affected_frames)),
    }
    classifier_info = {
        "name": resolved_name,
        "method": "image_ml",
        "model_type": "image_cnn",
        "input_modality": "image",
        "model_file": model_file,
        "cnn_backbone": getattr(model, "backbone_name", None),
        "model_input_height": int(model_height),
        "model_input_width": int(model_width),
        "metadata_version": CLASSIFICATION_METADATA_VERSION,
        "selection": selection_info,
        "overlap_handling": overlap_info,
    }

    def make_result(n_processed: int, n_failed: int) -> Dict[str, Any]:
        classifier_info["processing"] = {
            "n_total_events": int(n_events),
            "n_requested_events": int(requested_mask.sum()),
            "n_processed_events": int(n_processed),
            "n_excluded_by_mask": int(n_events - requested_mask.sum()),
            "n_excluded_by_overlap": int(n_overlap_excluded),
            "n_failed_events": int(n_failed),
        }
        return {
            "classifier_name": resolved_name,
            "class_map": class_map,
            "classifier": classifier_info,
        }

    if not np.any(inference_mask):
        return classes, probabilities, make_result(0, 0)

    if verbose:
        print(
            f"[3/4] Starting preprocessing and inference for "
            f"{int(inference_mask.sum()):,} events using {num_workers} worker(s)...",
            flush=True,
        )

    rtdc_path = _ds_get_rtdc_path(ds)
    tasks: List[Tuple[int, int]] = []
    for start_index, stop_index in _find_true_runs(inference_mask):
        tasks.extend(_split_run(start_index, stop_index, preprocess_chunk_size))

    preprocessing_failures = 0
    model.eval()

    def infer_and_store(
        indices: np.ndarray,
        images: np.ndarray,
        scales: np.ndarray,
    ) -> None:
        image_tensor = torch.from_numpy(images).to(torch.float32)[:, None, :, :] / 255.0
        image_tensor = ((image_tensor - 0.5) / 0.5).to(
            device,
            non_blocking=True,
        )
        scale_tensor = torch.from_numpy(scales).to(device, non_blocking=True)
        with torch.inference_mode(), _autocast_ctx(device, bool(use_amp)):
            logits = model(image_tensor, scale_tensor)
            batch_probabilities = F.softmax(logits, dim=1)
            confidence, predicted = torch.max(batch_probabilities, dim=1)
        output_indices = indices.astype(np.int64)
        classes[output_indices] = (
            predicted.detach().cpu().numpy().astype(np.int32) + 1
        ).astype(np.uint8)
        probabilities[output_indices] = (
            confidence.detach().cpu().numpy().astype(np.float32)
        )

    if num_workers == 0:
        buffered_indices: List[int] = []
        buffered_images: List[np.ndarray] = []
        buffered_scales: List[float] = []

        def flush_buffer() -> None:
            if not buffered_indices:
                return
            infer_and_store(
                np.asarray(buffered_indices, dtype=np.int32),
                np.stack(buffered_images, axis=0).astype(np.uint8, copy=False),
                np.asarray(buffered_scales, dtype=np.float32),
            )
            buffered_indices.clear()
            buffered_images.clear()
            buffered_scales.clear()

        task_iterator = (
            tqdm(tasks, desc="Preprocess+infer", unit="chunk")
            if show_progress
            else tasks
        )
        for start_index, stop_index in task_iterator:
            for event_index in range(int(start_index), int(stop_index)):
                try:
                    image = preprocessing_function(
                        ds,
                        event_index,
                        extra_pixels=extra_pixels,
                        **preprocess_kwargs,
                    )
                    if image is None:
                        raise ValueError("preprocessing returned None")
                    image, scale_x = _prepare_inference_image(
                        image,
                        model_height,
                        model_width,
                        use_cv=True,
                    )
                except Exception:
                    preprocessing_failures += 1
                    continue

                buffered_indices.append(event_index)
                buffered_images.append(image)
                buffered_scales.append(float(scale_x))
                if len(buffered_indices) >= batch_size:
                    flush_buffer()
        flush_buffer()
    else:
        mp_context = mp.get_context("spawn") if os.name == "nt" else mp.get_context()
        input_queue = mp_context.Queue(maxsize=max(2, prefetch_batches))
        output_queue = mp_context.Queue(maxsize=max(2, prefetch_batches))
        workers = []
        for _ in range(num_workers):
            process = mp_context.Process(
                target=_classify_images_worker,
                args=(
                    rtdc_path,
                    input_queue,
                    output_queue,
                    preprocessing_function,
                    int(extra_pixels),
                    preprocess_kwargs,
                    int(model_height),
                    int(model_width),
                    int(cv_threads_per_worker),
                ),
                daemon=True,
            )
            process.start()
            workers.append(process)

        task_source = iter(tasks)
        n_inflight = 0

        def submit_one() -> bool:
            nonlocal n_inflight
            try:
                task = next(task_source)
            except StopIteration:
                return False
            input_queue.put(task)
            n_inflight += 1
            return True

        while n_inflight < prefetch_batches and submit_one():
            pass

        buffered_indices = []
        buffered_images = []
        buffered_scales = []
        progress = (
            tqdm(total=len(tasks), desc="Preprocess+infer", unit="chunk")
            if show_progress
            else None
        )

        try:
            while n_inflight > 0:
                try:
                    result_item = output_queue.get(timeout=5.0)
                except queue.Empty:
                    failed_workers = [
                        process
                        for process in workers
                        if process.exitcode is not None and process.exitcode != 0
                    ]
                    if failed_workers:
                        exit_codes = [
                            process.exitcode for process in failed_workers
                        ]
                        raise RuntimeError(
                            "One or more image-preprocessing workers exited "
                            f"unexpectedly with exit codes {exit_codes}. Set "
                            "num_workers=0 to obtain the underlying error."
                        )
                    continue

                _, indices, images, scales, failed, worker_error = result_item
                n_inflight -= 1
                preprocessing_failures += int(failed)
                if progress is not None:
                    progress.update(1)
                if worker_error is not None:
                    raise RuntimeError(
                        f"Image preprocessing worker failed: {worker_error}"
                    )

                if indices is not None:
                    buffered_indices.extend(indices.tolist())
                    buffered_images.extend(list(images))
                    buffered_scales.extend(scales.tolist())

                while n_inflight < prefetch_batches and submit_one():
                    pass

                while len(buffered_indices) >= batch_size:
                    infer_and_store(
                        np.asarray(buffered_indices[:batch_size], dtype=np.int32),
                        np.stack(buffered_images[:batch_size], axis=0).astype(
                            np.uint8,
                            copy=False,
                        ),
                        np.asarray(buffered_scales[:batch_size], dtype=np.float32),
                    )
                    del buffered_indices[:batch_size]
                    del buffered_images[:batch_size]
                    del buffered_scales[:batch_size]

            if buffered_indices:
                infer_and_store(
                    np.asarray(buffered_indices, dtype=np.int32),
                    np.stack(buffered_images, axis=0).astype(
                        np.uint8,
                        copy=False,
                    ),
                    np.asarray(buffered_scales, dtype=np.float32),
                )
        finally:
            if progress is not None:
                progress.close()
            for _ in workers:
                try:
                    input_queue.put_nowait(None)
                except Exception:
                    break
            for process in workers:
                process.join(timeout=5)
            for process in workers:
                if process.is_alive():
                    process.terminate()

    n_processed = int(np.count_nonzero(probabilities > 0))
    if verbose:
        print(
            f"[3/4] Inference complete: {n_processed:,} events classified.",
            flush=True,
        )
        if preprocessing_failures:
            print(
                f"WARNING: {preprocessing_failures:,} events failed image "
                "preprocessing and remain class 0 with probability 0.",
                flush=True,
            )

    return classes, probabilities, make_result(
        n_processed,
        preprocessing_failures,
    )



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



def _classify_images_worker(
    rtdc_path: str,
    in_q: Any,
    out_q: Any,
    preprocessing_function: Callable[..., np.ndarray],
    extra_pixels: int,
    preprocess_kwargs: dict,
    H0: int,
    W0: int,
    cv_threads: int = 2,
) -> None:
    """Preprocess image chunks in a worker process for batched inference."""
    try:
        cv.setUseOptimized(True)
        cv.setNumThreads(int(cv_threads))
        use_cv = True
    except Exception:
        use_cv = False

    import dclab as worker_dclab

    with worker_dclab.new_dataset(rtdc_path) as ds:
        while True:
            task = in_q.get()
            if task is None:
                break
            start_idx, stop_idx = task
            task_len = int(stop_idx - start_idx)
            indices: List[int] = []
            images: List[np.ndarray] = []
            scales: List[float] = []
            failed = 0
            worker_error: Optional[str] = None

            try:
                for event_idx in range(int(start_idx), int(stop_idx)):
                    try:
                        image = preprocessing_function(
                            ds,
                            event_idx,
                            extra_pixels=extra_pixels,
                            **preprocess_kwargs,
                        )
                        if image is None:
                            raise ValueError("preprocessing returned None")
                        image, scale_x = _prepare_inference_image(
                            image,
                            H0,
                            W0,
                            use_cv=use_cv,
                        )
                    except Exception:
                        failed += 1
                        continue
                    indices.append(event_idx)
                    images.append(image)
                    scales.append(float(scale_x))
            except Exception as exc:
                worker_error = f"{type(exc).__name__}: {exc}"

            if indices:
                out_q.put((
                    task_len,
                    np.asarray(indices, dtype=np.int32),
                    np.stack(images, axis=0).astype(np.uint8, copy=False),
                    np.asarray(scales, dtype=np.float32),
                    failed,
                    worker_error,
                ))
            else:
                out_q.put((task_len, None, None, None, failed, worker_error))



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
class ImageImbalanceConfig:
    """
    Modular imbalance handling. Default is WeightedRandomSampler.
    Later you can swap strategy (e.g., class-weighted loss, focal loss, etc.).
    """
    strategy: str = "weighted_sampler"
    sampler_power: float = 1.0  # 1.0 -> inverse frequency; <1 softens


# -----------------------------
# Backbones (torchvision only)
# -----------------------------
def _build_image_backbone(name: str, pretrained: bool) -> Tuple[nn.Module, int, Optional[str]]:
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
    # Image model: backbone + scalar width feature (scale_x)
    # -----------------------------
    class ImageCNNWithScale(nn.Module):
        def __init__(self, backbone: str, num_classes: int, H0: int, W0: int, pretrained: bool = True):
            super().__init__()
            self.backbone_name = str(backbone)
            self.H0 = int(H0)
            self.W0 = int(W0)

            self.feature_extractor, feat_dim, self.weights_name = _build_image_backbone(backbone, pretrained=pretrained)
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
    # Image data sources: ZIP archive or folder
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


    class PreprocessedImageDataset(Dataset):
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


    def _make_sampler(labels: Sequence[int], cfg: ImageImbalanceConfig) -> WeightedRandomSampler:
        counts = Counter(labels)
        weights = [1.0 / (counts[int(y)] ** cfg.sampler_power) for y in labels]
        return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


    def _evaluate_image_model_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> Dict[str, Any]:
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
        imbalance_cfg: Optional[ImageImbalanceConfig] = None,
        seed: int = 123,
        num_workers: int = 0,
        pretrained: bool = True,
        cache_in_ram: bool = False,
    ) -> str:
        _print_runtime_info()
        os.makedirs(training_results_dir, exist_ok=True)
        imbalance_cfg = imbalance_cfg or ImageImbalanceConfig()

        classes = _infer_classes_from_zip(path_to_training_set, img_ext) if zip else _infer_classes_from_folder(path_to_training_set)

        full_ds = PreprocessedImageDataset(
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

        val_ds_full = PreprocessedImageDataset(
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
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(int(num_workers) > 0),
            prefetch_factor=_pf(),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(int(num_workers) > 0),
            prefetch_factor=_pf(),
        )

        device = _device()
        model = ImageCNNWithScale(backbone=cnn_backbone, num_classes=len(classes), H0=H0, W0=W0, pretrained=pretrained).to(device)

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

            val_metrics = _evaluate_image_model_epoch(model, val_loader, criterion, device=device)
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
                "format": "rtdc_tools_image_model",
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

        final_val = _evaluate_image_model_epoch(model, val_loader, criterion, device=device)
        cm = confusion_matrix(final_val["y_true"], final_val["y_pred"], labels=list(range(len(classes))))

        # Root-level artifacts
        _save_accuracy_plot(history, os.path.join(training_results_dir, f"{model_name}_acc.png"))
        _write_history_tsv(history, os.path.join(training_results_dir, f"{model_name}_history.tsv"))
        _save_confusion_matrix_percent(cm, classes, os.path.join(training_results_dir, f"{model_name}_confusion.png"))
    
        model_filename = f"{model_name}_" + "_".join(classes) + ".pt"
        model_path = os.path.join(training_results_dir, model_filename)

        bundle = {
            "format": "rtdc_tools_image_model",
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




def load_image_model(model_path: str) -> Tuple[nn.Module, List[str], int, int]:
    """Load a saved image-model bundle and reconstruct its CNN."""
    bundle = torch.load(model_path, map_location=_device())
    required = {"classes", "H0", "W0", "backbone", "model_state_dict"}
    missing = sorted(required.difference(bundle))
    if missing:
        raise KeyError(f"Image-model bundle is missing keys: {missing}")

    classes = [str(name) for name in bundle["classes"]]
    H0 = int(bundle["H0"])
    W0 = int(bundle["W0"])
    backbone = str(bundle["backbone"])
    model = ImageCNNWithScale(
        backbone=backbone,
        num_classes=len(classes),
        H0=H0,
        W0=W0,
        pretrained=False,
    ).to(_device())
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    return model, classes, H0, W0






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
    """Classify an RTDC file with an image model and write the result.

    The dataset-level classifier performs inference. This wrapper creates the
    output path, allocates class and probability features, registers classifier
    metadata, and writes the basin-linked RTDC file.
    """
    input_path = Path(path_to_rtdc)
    output_path = core._make_output_rtdc_path(input_path, output_suffix)

    if output_path.exists() and not overwrite:
        if verbose:
            print(f"[SKIP] File already exists: {output_path}", flush=True)
        return str(output_path)

    with dclab.new_dataset(str(input_path)) as ds:
        classes, probabilities, result = classify_ds_images_ml(
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
        metadata = ds.config.as_dict(pop_filtering=True)

    class_feature, probability_feature = core._allocate_userdef_features(
        str(input_path),
        2,
    )
    if verbose:
        print(
            f"Image classifier '{result['classifier_name']}' will store classes "
            f"in '{class_feature}' and probabilities in '{probability_feature}'.",
            flush=True,
        )
        print(f"[4/4] Writing classified RTDC file: {output_path}", flush=True)

    metadata = _register_classifier_metadata(
        metadata,
        class_feature=class_feature,
        probability_feature=probability_feature,
        class_map=result["class_map"],
        classifier=result["classifier"],
    )
    save_features_to_rtdc(
        basin_rtdc_path=str(input_path),
        output_rtdc_path=str(output_path),
        feature_dict={
            class_feature: classes,
            probability_feature: probabilities,
        },
        metadata=metadata,
        overwrite=True,
    )

    if verbose:
        print(f"[4/4] Finished: {output_path}", flush=True)
    return str(output_path)

