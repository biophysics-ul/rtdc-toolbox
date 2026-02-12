# ml_tools

Some additional info for the module `ml_tools.py`.

### Overview
The `ml_tools.py` module is designed to streamline the process of image classification in flow cytometry and other biological research. Its two key functionalities are:
1. Model Training: Train a modified ResNet18 model with user-provided training set.
2. Classification: Use the trained model to classify images and save the results. 

**Notes:** 
- The images in the training and sample sets do not need to be the same size. However, if you preprocess the training set images, you must apply the same preprocessing to the sample set images you wish to classify.
- See `ml_tools_example_script.py` in the `scripts` folder for a usage example.

## 1. Model Training
`train_model(training_set_path, output_model_folder)`

The `train_model` function in `ml_tools.py` handles the training of the cell classification model. It processes training set images stored in a ZIP file, trains the model, and saves the trained model along with performance metrics into `output_model_folder`. 

he function works as-is, but users can fine-tune training by modifying the following parameters:

- `EPOCHS`: Number of complete passes (epochs) through the training dataset.
- `BATCH_SIZE`: Number of training examples processed before the model updates its parameters.
- `VALIDATION_SPLIT`: Percentage of data used for validation.
- `PATIENCE_TOTAL`: Stops training if validation loss does not improve for a set number of epochs.
- `LEARNING_RATE`: Step size at each iteration during loss function optimization.
- `WEIGHT_DECAY`: Helps reduce overfitting by penalizing complex models.
- `DROPOUT_PROB`: Dropout probability, used to regularize the model and prevent overfitting.


**Notes:** 
- It is highly recommended to run the training on a GPU. Although it can run on a CPU, it will be significantly slower.

 ### Input data: Training set 
The training set must be stored in a single ZIP file, with images organized into folders named after their respective classes:

```
TrainingSet.zip  
├── class1/  
│   ├── image001.tif  
│   ├── image002.tif  
│   └── ...
├── class2/
│   ├── image001.tif
│   ├── image002.tif
│   └── ...
├── class3/
│   ├── image001.tif
│   ├── image002.tif
│   └── ...
└── ...
```

### Output data:
The trained model and some performance metrics are stored into `output_model_folder`.

- The filename of the final model (e.g., the model trained after the last training pass - epoch ) will have the structure "ml_model_{date}_classes_{classes}.pt", where {date} is the current date, and {classes} are the names of the classes inferred from the ZIP file, e.g.,   
    `ml_model_2025_05_25_classes_class1_class2_class3_class4.pt` 
- Performance metrics include:
    - The confusion matrix data and figure.
    - Per-class and overall performance data and figures.
    - Intermediate models and performance data for each epoch, saved in separate folders.

**Notes:** 
- You may rename the first part of the model filename, but you must keep the classes section unchanged. For example:
`my_model_classes_class1_class2_class3_class4.pt`  
This is because the classification process then inferes class names and their order from the model filename. 

## 2. Classification (inference)
`classify_images(sample_images_path, model_path, output_data_file_path)`

The `classify_images` function in `ml_tools.py` allows users to classify experimental sample images using a pre-trained model. It reads images from the ZIP file specified in sample_images_path, performs inference using the model specified in `sample_images_path`, performs uns inference using the model specified in the `model_path`, and saves the classification result a `tsv` file specified in the `output_data_file_path`. The result file is a table with three columns:  "img_name", "img_class" and "confidence". 


**Notes:**
- Classification runs much faster on a GPU, but with some patience, you can do it also on a CPU.
- Images in the `sample_images_path` have to be preprocessed in the same way as the images from the training set. 
- The filename of the model has to follow the structure "ml_model_{date}_classes_{classes}.pt", as described above.

## 3. Additional useful functions
- Once you classify your images using `classify_images`, you may want to view the results by extracting and sorting the images according to their predicted classes. For this, use the function `sort_class_images_from_zip(sample_images_path, classification_df, output_folder)` from  `ml_tools.py`.

    This function reads the images from the ZIP file specified in the `sample_images_path` and saves them in separate ZIP files in `output_folder`, organized by class. The classification data must be provided as a pandas DataFrame in `classification_df`. See `ml_tools_example_script.py` in the `scripts` folder for a usage example.
    


