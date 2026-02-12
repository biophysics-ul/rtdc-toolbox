# ML Cell Image Classification Framework for RT_DC

------------------------------------------------------------------------

## 1. Purpose and Scope

This is a Python module for AI-based training and classification of cell images in `.rtdc` files obtained by deformability cytometry (DC). To make it as robust and independent as possible, it operates only on segmented images of cells and does not take into account additional features typically measured in DC, such as deformation, cell size, porosity, etc. More information on DC can be found on the [dclab
homepage](https://dclab.readthedocs.io/en/stable/).

**It is designed for:**

- Training from manually curated datasets exported from `.rtdc` files
- Classification of images in `.rtdc` files, writing predictions to a new `.rtdc` file referring to the original `.rtdc` basin
- Efficient execution on a mid-range desktop (8-core CPU + RTX 4060 GPU)

**Core design decisions:**

- Images from the `.rtdc` file can be segmented and preprocessed using a custom function defined by the user (the same preprocessing should be used for building the training set and inference). For example, preprocessing may include background subtraction, cropping, padding, etc.
- Images are grayscale (`uint8` format)
- All images need to have the same height, but image width can vary
- Because PyTorch requires fixed-size inputs, images are rescaled in the x direction to W0 × H0 inside the dataset/loader, and the corresponding scaling factor is fed into the model together with the rescaled image (default W0 = 160) to preserve physical size information ((rescaled_image, scale_x)).  
- Minimal augmentation is used during training (brightness_jitter and contrast_jitter)
- CNN backbones with adaptive pooling from torchvision are be used as classification framework.  

**Our experience:**

- We tested the framework with `.rtdc` files recorded with Naiad (www.rivercyte.com) and segmented with ChipStream ML-based segmentation.  
- We find that `"convnext_tiny"` classifies our blood samples significantly better than `"resnet18"`, `"resnet50"` or `"efficientnet_b0"`. AI-tools mention that ConvNeXt-V2 or Swin-T could work even better, but we didn't test them.
- On our 8-core i7 CPU / GTX 3060 desktop PC, model training takes approx 20 min/epoch (with simple backbones such as restnet) and 60 min/epoch (with `"convnext_tiny"`). Classificiation of a typical 10-min recording takes approx 20 min.  

## 2. Training

### Training Set

Training images must be manually classified and stored in subfolders. Each folder name represents a class label. The folders can be optionally stored in a single ZIP file.  

    TrainingSet/
      RBC/
        img001.png
        ...
      platelets/
        img010.png
        ...
      WBC/
        img020.png
        ...

*The easiest* way to build a training set is to export cell images from rtdc files using function `extract_preprocessed_images_from_rtdc_to_zip.py` and the same image preprocessing that will be then used for image classification. The images can be then manually sorted into folders using standard Windows Explorer (setting view to "Large icons").  

### Running the Training

Once the training set is ready, training is started with  

    train_model(
        path_to_training_set,
        training_results_dir,
        model_name,
        cnn_backbone,
        ...
        )

The main parameters are:

- `training_results_dir`: where to store the results
- `model_name`: filename of the model, e.g., `"MyModel"`
- `cnn_backbone`: CNN backbone used for training, e.g., `"resnet18"` (default), `"resnet50"`, `"efficientnet_b0"`, `"convnext_tiny"`, etc.
- `n_epochs`: number of training epochs (training rounds). Default is 15.

### Training Results

Training results are stored in the specified folder. When training is complete, the folder will contain:

- The final trained model  
- An image showing training/validation accuracy versus epochs
- An image showing final confusion matrix
- Folders containing all intermediate models
- A JSON file with training metadata (the same metadata is stored also in the model file)

The model filename will encode the class labels defined by the subfolders in the training set, for example `MyModel_RBC_platelets_WBC.pt`, however, you can freely rename it later because all the metadata about the classes is stored inside the model.  

## 3. Classification

Classification is performed using:

    classify_rtdc_ml(
        path_to_rtdc,
        path_to_model, 
        class_feature,
        prob_feature,
        model_ext,
        ...
        )

Classification creates a new `.rtdc` file in the same folder as the original, with the same name and extra extension defined by `model_ext` parameter (default `"_class"`) to distinguish
it from the original file. Results are stored in RTDC features specified by `class_feature` and `prob_feature`:

- `class_feature`: feature storing the predicted class index (integer number corresponding to the classes defined in the model filename, e.g., RBC=0, platelets=1, WBC=2, ...). Default: `"userdef1"`
- `prob_feature`: feature storing the probability of the predicted class. Default: `"userdef2"`

## 4. Preprocessing

The preprocessing function is invoked as:

    img = preprocess_image(ds, img_indx, **kwargs)

The function must return a grayscale uint8 image of fixed height. The same preprocessing function must be used consistently for training and inference.