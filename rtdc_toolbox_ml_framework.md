# ML Cell Image Classification Framework for RT-DC

rtdc-toolbox framework version 2.0 (August 2026)

## 1. Purpose and scope

The image-model framework trains and applies CNN classifiers to segmented grayscale cell images from RT-DC files. It is intentionally image-only: deformation, area, brightness, texture, and other scalar RT-DC features are not passed to the CNN.

The framework supports:

- training from manually curated image folders or a ZIP archive;
- custom image preprocessing before training-set export and RT-DC inference;
- torchvision backbones with a separate scalar width input;
- masked classification of selected RT-DC events;
- optional exclusion of same-frame events whose horizontal extents overlap;
- GPU inference when CUDA is available;
- basin-linked RT-DC outputs with automatically allocated class and probability features;
- metadata that records class names, feature locations, model details, selection, and processing counts.

The PyTorch implementation is in `rtdc_toolbox_ml.py`. Public functions are also exposed through lazy wrappers in `rtdc_toolbox.py`, so normal scripts can use:

```python
import rtdc_toolbox
```

and call:

```python
rtdc_toolbox.train_model_on_images(...)
rtdc_toolbox.classify_rtdc_images_ml(...)
```

## 2. Image representation

### Grayscale images

Preprocessed images must be two-dimensional grayscale arrays with dtype `uint8`.

### Training height and variable width

All images in one training dataset must have the same height. The height of the first image becomes the model input height `H0`.

Image width may vary. For each image:

1. the original width is divided by the configured model width `W0` to obtain `scale_x`;
2. the image is resized to `H0 × W0`;
3. the CNN receives the resized image;
4. the final classifier layer also receives `scale_x`.

This preserves one scalar measure of the original horizontal extent after the image itself has been normalized to a fixed size.

### Inference height handling

At inference time, the preprocessing function may return an image whose height differs from the model's `H0`. The framework then:

1. resizes the image to height `H0` while preserving its aspect ratio;
2. computes `scale_x` from the width after height normalization;
3. resizes the width to `W0`.

Using the same preprocessing method for training-set generation and inference remains essential.

## 3. Preparing a training set

Training images must be placed in one top-level folder per class:

```text
TrainingSet/
    RBC/
        img001.png
        img002.png
    platelets/
        img010.png
    WBC/
        img020.png
```

The same structure may be stored in a ZIP archive:

```text
TrainingSet.zip
    RBC/img001.png
    RBC/img002.png
    platelets/img010.png
    WBC/img020.png
```

Class names are read from folder names and sorted alphabetically before class indices are assigned. The class order is stored inside every model bundle, so the model file may be renamed later without losing class information.

### Exporting images from RT-DC files

Use:

```python
import rtdc_toolbox

rtdc_toolbox.export_preprocessed_images_from_rtdc_to_zip(
    rtdc_path=r"D:\RTDC\Sample_dcn.rtdc",
    zip_path=r"D:\RTDC\Training\UnsortedImages.zip",
    mask=None,
    extra_pixels=15,
    no_overlap=True,
    img_ext=".png",
    n=10_000,
    random_images=True,
    saving_mode="overwrite",
)
```

The exporter uses `rtdc_toolbox.preprocess_rtdc_image`. Exported images can then be sorted manually into class folders. Use `saving_mode="overwrite"` to replace an existing ZIP or `saving_mode="append"` to add images to it.

The training function does not call the preprocessing function itself. It assumes that all training images have already been preprocessed.

## 4. Training

A minimal training call is:

```python
import rtdc_toolbox

model_path = rtdc_toolbox.train_model_on_images(
    path_to_training_set=r"D:\RTDC\Training\TrainingSet.zip",
    training_results_dir=r"D:\RTDC\Training\Results",
    model_name="BloodClassifier",
    zip=True,
    cnn_backbone="convnext_tiny",
    img_ext=".png",
    n_epochs=15,
    W0=160,
    val_split=0.2,
    batch_size=64,
    lr=3e-4,
    weight_decay=1e-4,
    brightness_jitter=0.0,
    contrast_jitter=0.0,
    seed=123,
    num_workers=0,
    pretrained=True,
    cache_in_ram=False,
)
```

See `example_train_image_model.py` for a complete editable example.

### Main parameters

- `path_to_training_set` — ZIP archive or directory containing class folders.
- `training_results_dir` — output directory for models and training reports.
- `model_name` — base name used for reports and metadata.
- `zip` — `True` for a ZIP archive and `False` for a directory tree.
- `cnn_backbone` — one of `resnet18`, `resnet50`, `efficientnet_b0`, or `convnext_tiny`.
- `n_epochs` — number of training epochs.
- `W0` — fixed model input width.
- `val_split` — fraction of samples used for validation.
- `batch_size` — training batch size.
- `brightness_jitter`, `contrast_jitter` — optional minimal augmentation.
- `num_workers` — PyTorch data-loader workers.
- `cache_in_ram` — cache decoded images in memory.
- `pretrained` — initialize the torchvision backbone with pretrained weights.

On Windows, `num_workers=0` is the safest setting. When `cache_in_ram=True`, using `num_workers=0` also avoids duplicating the entire cache in worker processes.

### Class imbalance

By default, training uses `ImageImbalanceConfig(strategy="weighted_sampler", sampler_power=1.0)`. This applies inverse-frequency weighted sampling. A smaller `sampler_power` softens the weighting.

### Training outputs

The results directory contains:

- one `epoch_NNN/` folder per epoch;
- `epoch_NNN/model.pt` checkpoint bundles;
- per-epoch confusion matrices and `meta.json`;
- `<model_name>_history.tsv`;
- `<model_name>_acc.png`;
- `<model_name>_confusion.png`;
- `<model_name>_meta.json`;
- the final best model, named `<model_name>_<class1>_<class2>_....pt`.

The final model bundle stores:

- CNN backbone;
- class names and order;
- `H0` and `W0`;
- normalization description;
- imbalance settings;
- model state dictionary.

## 5. Loading a trained model

```python
model, classes, H0, W0 = rtdc_toolbox.load_image_model(model_path)
```

The model is reconstructed on CUDA when available, otherwise on CPU.

## 6. Dataset-level inference

`classify_ds_images_ml` works on an already-open `dclab` dataset:

```python
classes, probabilities, result = rtdc_toolbox.classify_ds_images_ml(
    ds,
    mask=None,
    path_to_model=model_path,
    preprocessing_function=rtdc_toolbox.preprocess_rtdc_image,
    classifier_name=None,
    mask_description=None,
    verbose=True,
    show_progress=True,
    extra_pixels=15,
    batch_size=1024,
    num_workers=0,
)
```

The function:

- returns full-length class and probability arrays;
- does not allocate `userdefN` features;
- does not modify metadata;
- does not write a file.

Class values are encoded as:

- `0` — unclassified or not processed;
- `1..N` — model classes in the order stored in the model bundle.

Events outside `mask`, excluded overlaps, and preprocessing failures remain class `0` with probability `0`.

The returned `result` dictionary contains the class map and classifier metadata.

## 7. RT-DC file classification

Use the file-level wrapper for normal classification:

```python
output_rtdc = rtdc_toolbox.classify_rtdc_images_ml(
    path_to_rtdc=r"D:\RTDC\Sample_dcn.rtdc",
    path_to_model=model_path,
    preprocessing_function=rtdc_toolbox.preprocess_rtdc_image,
    classifier_name="BloodClassifier",
    mask=None,
    mask_description=None,
    output_suffix="_images_ml",
    overwrite=False,
    verbose=True,
    show_progress=True,
    extra_pixels=15,
    batch_size=1024,
    num_workers=0,
    prefetch_batches=8,
    preprocess_chunk_size=256,
    use_amp=None,
    preprocess_kwargs=None,
    cv_threads_per_worker=2,
    exclude_x_overlapping_events=True,
    min_x_overlap=0.0,
    min_x_overlap_fraction=0.0,
)
```

The wrapper:

1. opens the input RT-DC file;
2. runs `classify_ds_images_ml`;
3. allocates the next two free `userdefN` features;
4. writes class labels and probabilities;
5. registers metadata format `2.0`;
6. writes a new basin-linked RT-DC file;
7. returns its path.

`output_suffix` is inserted before `.rtdc`. With the default suffix:

```text
Sample_dcn.rtdc
Sample_dcn_images_ml.rtdc
```

`overwrite=False` is the safe default.

## 8. Masks

The `mask` argument may contain one Boolean value per RT-DC event. Only `True` events are requested for inference.

A mask can be generated from an earlier classifier:

```python
wbc_mask = rtdc_toolbox.make_classifier_mask(
    path_or_ds=classified_rtdc,
    classifier_name="default",
    classes="WBC",
    probability_threshold=0.95,
    probability_comparison=">",
)
```

The optional `mask_description` is stored in classifier metadata.

### Passing an image-selected WBC mask to Kaliman25

See `example_Kaliman25_classification.py` for a standalone Kaliman workflow that can switch between the original whole-blood Step 1 and image-selected WBCs.

When an image classifier has already isolated WBCs, the same mask can be passed to Kaliman25 with `preselected_wbcs=True`:

```python
kaliman_rtdc = rtdc_toolbox.classify_rtdc_WBC_Kaliman25(
    path_to_rtdc=original_rtdc,
    mask=wbc_mask,
    mask_description="WBC events selected by image classifier",
    preselected_wbcs=True,
    output_suffix="_kaliman25",
)
```

This skips only Kaliman Step 1, the crude WBC prefilter based on area, deformation, and area ratio. The mask-selected events are used directly as the input to Step 2; all later Kaliman filters and clustering steps remain unchanged. The default is `preselected_wbcs=False`, which preserves the original Kaliman workflow.

## 9. Same-frame overlap exclusion

With `exclude_x_overlapping_events=True`, the framework checks scalar RT-DC features:

- `frame`;
- `pos_x`;
- `size_x`.

Each event is represented by:

```text
[pos_x - size_x/2, pos_x + size_x/2]
```

Both events in a qualifying same-frame overlap remain class `0` with probability `0`.

Controls:

- `min_x_overlap` — minimum overlap width in µm;
- `min_x_overlap_fraction` — minimum overlap divided by the narrower event width.

In dclab, both `pos_x` and `size_x` are expressed in µm, so no pixel-to-µm conversion is needed for this classifier-level overlap check. The check does not read contours. This is distinct from `preprocess_rtdc_image(no_overlap=True)`: that preprocessing option uses contour x-coordinates in image pixels only to limit the image crop.

## 10. Preprocessing function

The preprocessing function is called as:

```python
image = preprocessing_function(
    ds,
    event_index,
    extra_pixels=extra_pixels,
    **preprocess_kwargs,
)
```

It must return a two-dimensional grayscale `uint8` array or `None`.

The default function is:

```python
rtdc_toolbox.preprocess_rtdc_image
```

It performs background subtraction and x-direction cropping around the event contour.

Any custom function used for inference should also be used to generate the training images.

## 11. Performance controls

- `batch_size` controls GPU/CPU inference batch size.
- `num_workers=0` performs preprocessing in the main process and is the safest Windows/Spyder setting.
- `num_workers>0` starts preprocessing worker processes.
- `prefetch_batches` limits queued preprocessing chunks.
- `preprocess_chunk_size` controls how many contiguous events are assigned to one preprocessing task.
- `cv_threads_per_worker` limits OpenCV threads inside each worker.
- `use_amp=None` enables automatic mixed precision when CUDA is used.
- `show_progress` controls progress bars.
- `verbose` controls informational messages.

GPU inference still works with `num_workers=0`; only image preprocessing remains single-process.

## 12. Metadata and reports

After classification, inspect metadata with:

```python
rtdc_toolbox.print_rtdc_classifier_metadata(output_rtdc)
```

Create reports with:

```python
rtdc_toolbox.export_rtdc_class_stability_report(
    path_to_rtdc=output_rtdc,
    classifier_name="default",
)

rtdc_toolbox.export_rtdc_class_examples(
    path_to_rtdc=output_rtdc,
    classifier_name="default",
    report_ext="_classes",
    preprocessing_function=rtdc_toolbox.preprocess_rtdc_image,
    prob_thresh=0.95,
)
```

The report functions resolve the class and probability features from metadata instead of assuming fixed `userdefN` locations.

## 13. Practical experience

The framework has been used with Naiad RT-DC files segmented by ChipStream. In the authors' blood-cell datasets, `convnext_tiny` performed better than the tested `resnet18`, `resnet50`, and `efficientnet_b0` models. Performance and training time depend strongly on hardware, image count, backbone, batch size, storage speed, and preprocessing.
