# rtdc-toolbox

`rtdc-toolbox` is a small Python toolkit for working with RT-DC (`.rtdc`) files produced by deformability cytometry. It uses [`dclab`](https://dclab.readthedocs.io/en/stable/) for RT-DC files, scikit-learn for the Kaliman25 WBC classifier, and PyTorch/torchvision for image-model training and inference.

rtdc-toolbox framework version 2.0 (August 2026)

## Repository files

- `rtdc_toolbox.py` — torch-free RT-DC utilities, Kaliman25 classification, metadata helpers, reports, and lazy wrappers for image-model functions.
- `rtdc_toolbox_ml.py` — PyTorch image-model training and inference.
- `rtdc_viewer.py` — PyQt5 viewer with classifier/class selection, event inspection, and copyable scalar feature values.
- `rtdc_image_viewer_openCV.py` — simpler OpenCV viewer for RT-DC images, contours, class labels, and probabilities.
- `example_train_image_model.py` — example for training an image classifier.
- `example_export_random_images_by_class.py` — example for exporting random high-confidence images from classified RT-DC files.
- `example_image_classification.py` — example for batch image-based classification and image-classifier reports.
- `example_Kaliman25_classification.py` — example for Kaliman25 classification, either with its original Step 1 WBC selection or with image-selected WBCs.
- [`rtdc_toolbox_ml_framework.md`](rtdc_toolbox_ml_framework.md) — detailed description of the image-model workflow.

## Typical workflow and output names

A typical ChipStream workflow may produce files such as:

- `MySample.rtdc` — original RT-DC file.
- `MySample_dcn.rtdc` — file segmented with ChipStream.
- `MySample_dcn_images_ml.rtdc` — image-classified output using the default `output_suffix="_images_ml"`.
- `MySample_dcn_kaliman25.rtdc` — Kaliman25-classified output using the default `output_suffix="_kaliman25"`.
- `MySample_dcn_images_ml.png` — class-stability report.
- `MySample_dcn_images_ml_classes.png` — class-example report.

The output suffix is configurable, so names such as `_image_b7` are also valid.

Classifiers write their results to the next free scalar RT-DC features, normally `userdefN`. The exact class and probability features are recorded in classifier metadata and should not be assumed to be `userdef1` and `userdef2`.

## Classifier metadata

Classifier metadata uses registry format `2.0` and is stored in the RT-DC user metadata under `rtdc_tools_classification_json`.

Each classifier entry records:

- a unique classifier name;
- the feature containing class labels;
- the optional feature containing probabilities;
- the map from integer class values to class names;
- classifier-specific details such as method, model file, CNN backbone, event selection, and processing counts.

Class value `0` is reserved for `unclassified`. It is also used for events that were not processed, for example events outside a mask or image events excluded because preprocessing failed.

Use these helpers instead of hardcoding `userdefN` locations:

```python
import rtdc_toolbox

all_metadata = rtdc_toolbox.extract_rtdc_classifier_metadata(path_to_rtdc)
one_classifier = rtdc_toolbox.get_classifier_metadata(
    path_to_rtdc,
    classifier_name="default",
)
class_maps = rtdc_toolbox.get_classes_from_rtdc(path_to_rtdc)
rtdc_toolbox.print_rtdc_classifier_metadata(path_to_rtdc)
```

`classifier_name="default"` selects the first classifier in metadata insertion order. A concrete classifier name selects it explicitly.

## Main usage

### 1. View RT-DC images

Run:

```text
rtdc_viewer.py
```

The PyQt5 viewer:

- opens an RT-DC file selected through a file dialog;
- displays images in a grid;
- toggles contours with Space when contours are present;
- reads class names and feature locations from classifier metadata;
- falls back to `userdef1`/`userdef2` for legacy files without the new metadata;
- filters available classes with keys `0`–`9`;
- resets the class filter with Backspace or Delete;
- opens a clicked event in a separate window with image, background image, and copyable scalar feature values.

The simpler `rtdc_image_viewer_openCV.py` is also retained. The PyQt5 viewer lets you select the classifier from a dropdown instead of using a fixed `CLASSIFIER_NAME`.

### 2. Train an image classifier

See:

```text
example_train_image_model.py
```

The public training function is:

```python
rtdc_toolbox.train_model_on_images(...)
```

Training images must already be preprocessed and arranged in class folders, either in a directory or a ZIP archive.

### 3. Classify images

See:

```text
example_image_classification.py
```

The example recursively finds segmented RT-DC files without an image-classified analogue, runs the image classifier, prints classifier metadata, and exports class-stability and class-example reports.

With `exclude_x_overlapping_events=True`, image classification checks same-frame horizontal overlap using the scalar features `pos_x` and `size_x`. In dclab both are in µm, so `min_x_overlap` is also in µm. This is separate from `preprocess_rtdc_image(no_overlap=True)`, where contour coordinates are image pixels and are used only to limit the image crop.

### 4. Run Kaliman25

See:

```text
example_Kaliman25_classification.py
```

The Kaliman example has one switch:

```python
USE_IMAGE_SELECTED_WBCS = True
```

With `False`, Kaliman25 runs on whole blood and applies its original Step 1 crude-WBC prefilter. With `True`, the script reads a WBC mask from the corresponding image-classified RT-DC file and calls Kaliman25 with `preselected_wbcs=True`. This bypasses only Step 1; Step 2 and all later subtype-classification steps remain unchanged.

### 5. Export quality-control reports

```python
rtdc_toolbox.export_rtdc_class_stability_report(
    path_to_rtdc=classified_rtdc,
    classifier_name="default",
    features_to_plot=("area_um", "deform"),
)

rtdc_toolbox.export_rtdc_class_examples(
    path_to_rtdc=classified_rtdc,
    classifier_name="default",
    report_ext="_classes",
)
```

The stability report plots class percentages and selected feature statistics in moving event windows. The class-example report exports representative preprocessed images for each observed class.

## Public function overview

### Image preparation and export

#### `get_img_diff`

Computes a clipped, shifted difference between two `uint8` images. The default shift places unchanged background near 128.

#### `preprocess_rtdc_image`

Background-subtracts and crops one RT-DC event image around its contour. With `no_overlap=True`, crop borders are limited by neighboring events in the same frame.

#### `export_preprocessed_images_from_rtdc_to_zip`

Exports preprocessed event images from an RT-DC file to a ZIP archive for manual curation and model training. `saving_mode="overwrite"` replaces an existing ZIP; `saving_mode="append"` adds images to it.

### Image-model training and loading

#### `train_model_on_images`

Trains an image-only CNN from a ZIP archive or directory containing one class folder per label. It returns the path to the best final model bundle and writes accuracy, confusion-matrix, history, checkpoint, and metadata files.

#### `load_image_model`

Loads a model bundle created by `train_model_on_images` and returns the model, class names, model input height, and model input width.

### Dataset-level classification

Dataset-level functions work on an already-open `dclab` dataset. They do not allocate RT-DC features, register metadata, or write files.

#### `classify_ds_images_ml`

Returns:

```python
classes, probabilities, result
```

Both arrays have one value per event. Events outside the mask, excluded overlaps, and preprocessing failures remain class `0` with probability `0`.

#### `classify_ds_WBC_Kaliman25`

Returns:

```python
classes, None, result
```

Events outside the mask remain class `0`. Kaliman25 does not produce probabilities. By default the original Kaliman Step 1 crude-WBC prefilter is applied. Set `preselected_wbcs=True` when the supplied mask already isolates WBCs; this skips only Step 1, while all later Kaliman steps remain unchanged. A mask is required in this mode.

### File-level classification

File-level functions create a new basin-linked RT-DC file, allocate the next free `userdefN` features, register metadata, and return the output path.

#### `classify_rtdc_images_ml`

Classifies event images using a saved PyTorch model. The default output suffix is `_images_ml`.

#### `classify_rtdc_WBC_Kaliman25`

Runs the feature-based Kaliman25 WBC classifier. The default output suffix is `_kaliman25`. It accepts `preselected_wbcs=True` for workflows in which a preceding classifier has already isolated WBCs; in that mode Kaliman Step 1 is skipped.

The implementation follows:

Kaliman et al., *Automation and improvement of WBC mechanical profiling in deformability cytometry*, Biophysical Journal, 2025.  
DOI: `10.1016/j.bpj.2025.10.007`

### Masks and metadata

#### `make_classifier_mask`

Creates a Boolean event mask from class names or class values and, when available, an optional probability threshold.

#### `get_classes_from_rtdc`

Returns class maps for all registered classifiers:

```python
{
    "classifier_name": {
        0: "unclassified",
        1: "RBC",
        2: "WBC",
    }
}
```

#### `extract_rtdc_classifier_metadata`

Returns a summary containing all classifier entries.

#### `get_classifier_metadata`

Returns one classifier entry selected by name or by `"default"`.

#### `print_rtdc_classifier_metadata`

Prints all registered classifier metadata and returns the same summary dictionary.

### Reports and file writing

#### `export_rtdc_class_stability_report`

Creates a PNG report showing class percentages and selected feature means and standard deviations across moving event windows.

#### `export_rtdc_class_examples`

Creates a PNG containing representative preprocessed images from each class. Probability filtering is applied only when the selected classifier has a probability feature; class `0` is always sampled without a probability threshold.

#### `save_features_to_rtdc`

Writes supplied feature arrays and metadata to a new basin-linked RT-DC file.

## Installation

When `dclab` and PyTorch are already available, place `rtdc_toolbox.py` and `rtdc_toolbox_ml.py` in the same folder as your scripts or on the Python path.

To create a new environment:

1. Install PyTorch using the command recommended for your operating system and GPU on the [PyTorch installation page](https://pytorch.org/get-started/locally/).

2. Install the remaining dependencies:

```bash
pip install numpy opencv-python dclab tqdm matplotlib scikit-learn pillow
```

PyTorch is optional for the torch-free utilities and Kaliman25 classifier, but required for image-model training and inference.

Versions used successfully in the original development environment included:

```text
Python        3.13.2
numpy         2.2.6
opencv-python 4.11.0.86
dclab         0.64.0
tqdm          4.67.1
matplotlib    3.10.3
scikit-learn  1.6.1
torch         2.7.0+cu126
torchvision   0.22.0+cu126
```

## Notes and citation

`rtdc-toolbox` is based on:

Fajdiga Lebar et al., *Effects of intravenous lipid emulsions on Jurkat cells assessed using label-free deformability cytometry*, Scientific Reports, 2025.  
DOI: `10.1038/s41598-025-33582-7`

Please cite this paper when the toolbox contributes to published work.

The current implementation was rewritten with substantial assistance from AI and tested on real RT-DC data. It may still contain bugs, so results should be inspected and validated for each experimental workflow.

The original implementation is available in the `legacy` folder.

## Author

Jure Derganc  
Institute of Biophysics, Faculty of Medicine, University of Ljubljana
