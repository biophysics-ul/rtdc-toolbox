# rtdc-toolbox

`rtdc-toolbox` is a small Python toolkit for working with RT-DC (`.rtdc`) files produced by deformability cytometry. It uses [`dclab`](https://dclab.readthedocs.io/en/stable/) for reading and working with RT-DC files.

The toolkit is especially useful if you want to develop your own image-based event classifier. It provides functions for model training and event classification within a unified framework. For example, we have been using it to classify events in whole-blood samples into classes such as RBCs, WBCs, platelets, and debris.

*rtdc-toolbox framework version 2.0 (August 2026)*

## Table of contents

- [1. Repository files](#1-repository-files)
- [2. Typical classification workflow and output names](#2-typical-classification-workflow-and-output-names)
- [3. Classifier metadata](#3-classifier-metadata)
- [4. Main usage](#4-main-usage)
- [5. Main toolbox functions](#5-main-toolbox-functions)
- [6. Installation](#6-installation)
- [7. Citation](#7-citation)
- [8. Author](#8-author)

## 1. Repository files

- `rtdc_toolbox.py` — a module with utilities for working with RT-DC files.

- `rtdc_toolbox_ml.py` — a module for training image classifiers and applying them to RT-DC files. It requires PyTorch.

- `example_train_image_model.py` — an example script for training your own image classifier.

- `example_image_classification.py` — an example script for classifying events in RT-DC files using your image classifier.

- `example_Kaliman25_classification.py` — an example script for classifying WBCs into subgroups (lymphocytes, neutrophils, etc.) using the method described by Kaliman et al. (2025).

- `example_export_random_images_by_class.py` — an example script for exporting random images from classified RT-DC files; useful for inspecting and iteratively improving image training sets.

- `rtdc_viewer.py` — a simple viewer for images in RT-DC files, with classifier/class selection and event inspection.

- `rtdc_toolbox_ml_framework.md` — a more detailed description of the image-classification workflow.

## 2. Typical classification workflow and output names

A typical RT-DC ChipStream workflow may produce files such as:

- `MySample.rtdc` — the original RT-DC file.
- `MySample_dcn.rtdc` — a file segmented with ChipStream and linked to the original RT-DC file.

When you run an image classifier, it produces another RT-DC file containing the event-classification results, for example:

- `MySample_dcn_myext.rtdc` — this file is linked to `MySample_dcn.rtdc`. The suffix `_myext` is user-configurable and should reflect the model used for classification. For example, we use `_b7c` as the suffix for a model developed for whole blood that classifies events into seven distinct classes.

- `MySample_dcn_myext_Kaliman25.rtdc` — WBCs identified by the image-based classifier can be further classified into WBC subtypes (lymphocytes, neutrophils, etc.) using the classifier based on Kaliman et al. (2025).

**Important:**  
Classification also produces two images that help you inspect the quality of the experiment and the classification:

- `MySample_dcn_myext.png` — shows how selected features and class percentages change during the experiment. This is useful for checking experimental stability and identifying potential problems during data acquisition.

- `MySample_dcn_myext_classes.png` — contains example images of events from all classes and helps you visually assess whether the classification is performing as expected.

**Some additional notes**

- Classification information is stored directly in the RT-DC file as additional scalar features. Typically, the class number is stored in `userdef1` and the classification probability in `userdef2`.

- Classified RT-DC files can be analyzed with standard RT-DC software such as Cytoplot. For example, you can filter events using `userdef1` and `userdef2`, or use `userdef1` as the hue to color events by class in scatter plots.

## 3. Classifier metadata

In addition to the numerical class values stored in the RT-DC file, `rtdc-toolbox` also saves information describing the classifier that produced them. This includes the classifier name, the meaning of the class numbers, and, when available, the classification probabilities.

This information makes it possible for the toolbox and the viewers to recognize automatically which classes are present in a file. In most cases, you therefore do not need to remember which `userdef` feature contains the class number or probability.

Class `0` is reserved for events that are not classified. This may include events that were intentionally excluded from classification or events for which classification was not possible.

## 4. Main usage

### 4.1 View RT-DC images

Run:

```text
rtdc_viewer.py
```

The viewer lets you:

- open an RT-DC file;
- browse event images in a grid;
- show or hide event contours;
- select a classifier and display individual classes;
- see the predicted class and classification confidence;
- click an event to inspect its image, background image, and selected RT-DC features.

A simpler OpenCV-based viewer, `rtdc_image_viewer_openCV.py`, is also included.

### 4.2 Train your own image classifier

See:

```text
example_train_image_model.py
```

The script shows how to train an image classifier from a set of manually sorted cell images.

Training images should be organized into folders according to class, for example:

```text
TrainingSet/
    RBC/
    WBC/
    platelets/
    debris/
```

The training set can be stored either as a normal folder or as a ZIP archive.

During training, the toolbox creates model files and several plots that help you evaluate classification performance.

### 4.3 Classify events in RT-DC files

See:

```text
example_image_classification.py
```

This script applies a trained image classifier to RT-DC files.

For each event, the classifier assigns:

- a class;
- a classification probability.

Events that are excluded or cannot be classified remain unclassified.

The script can also exclude events that overlap horizontally with another event in the same camera frame. This helps avoid classifying images in which two cells may interfere with each other.

After classification, the script automatically creates summary plots and example images so that the results can be checked visually.

### 4.4 Classify WBC subtypes with the Kaliman25 method

See:

```text
example_Kaliman25_classification.py
```

The Kaliman25 classifier separates WBCs into subtypes using RT-DC measurements rather than cell images.

It can be used in two ways:

1. directly on whole-blood data, using the original WBC-selection procedure described in the Kaliman method;
2. after the image classifier has already identified which events are WBCs.

The second option is useful when the image classifier provides a cleaner initial selection of WBCs.

### 4.5 Export images for inspection or training

The toolbox can export preprocessed event images from RT-DC files into ZIP archives.

This is useful for:

- visually checking classification results;
- selecting examples for a training set;
- expanding an existing training set;
- manually correcting misclassified examples.

See:

```text
example_export_random_images_by_class.py
```

## 5. Main toolbox functions

Most users can work directly from the example scripts without calling individual functions manually. The main functions provided by the toolbox are summarized below.

### Image preparation

`preprocess_rtdc_image(...)`

Prepares an event image for training or classification. It performs background subtraction and crops the image around the event.

`export_preprocessed_images_from_rtdc_to_zip(...)`

Exports preprocessed images from RT-DC files into a ZIP archive.

### Image-classifier training

`train_model_on_images(...)`

Trains an image classifier from manually sorted training images.

`load_image_model(...)`

Loads a previously trained model.

### Image classification

`classify_rtdc_images_ml(...)`

Applies a trained image classifier to an RT-DC file and saves the classification results in a new RT-DC file.

### WBC classification

`classify_rtdc_WBC_Kaliman25(...)`

Classifies WBCs into subtypes using the method based on Kaliman et al. (2025).

### Class selection

`make_classifier_mask(...)`

Creates a selection of events belonging to one or more classifier classes. This is useful when one classification step is used as input for another analysis.

### Quality-control reports

`export_rtdc_class_stability_report(...)`

Creates a plot showing how class percentages and selected RT-DC measurements change during the experiment.

`export_rtdc_class_examples(...)`

Creates an image containing representative examples from each predicted class.

## 6. Installation

The toolbox consists mainly of the two modules:

```text
rtdc_toolbox.py
rtdc_toolbox_ml.py
```

Place them in the same folder as your analysis scripts, or somewhere on your Python path.

The main required packages are:

- `dclab`
- NumPy
- OpenCV
- Matplotlib
- scikit-learn

PyTorch and torchvision are additionally required for training and using image classifiers.

A typical installation of the non-PyTorch dependencies is:

```bash
pip install numpy opencv-python dclab tqdm matplotlib scikit-learn pillow
```

For PyTorch, use the installation instructions provided on the official PyTorch website, because the recommended version depends on your operating system and GPU.

Versions used successfully during development included:

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

## 7. Citation

`rtdc-toolbox` is based on:

Fajdiga Lebar et al., *Effects of intravenous lipid emulsions on Jurkat cells assessed using label-free deformability cytometry*, Scientific Reports, 2025.  
DOI: `10.1038/s41598-025-33582-7`

Please cite this paper when the toolbox contributes to published work.

The current implementation was rewritten with substantial assistance from AI and tested on real RT-DC data. It may still contain bugs, so classification results should always be inspected and validated for each experimental workflow.

The original implementation is available in the `legacy` folder.

## 8. Author

Jure Derganc  
Institute of Biophysics, Faculty of Medicine, University of Ljubljana
