# -*- coding: utf-8 -*-
"""
Created on Fri May 23 13:01:45 2025
@author: jderganc
"""

import ml_tools
import pandas as pd
import os


ml_tools.IMG_EXT='.tif'
ml_tools.EPOCHS=50

root_folder= r"D:\JurkatCellsInLipidEmulsions"
training_set_folder=os.path.join(root_folder,"TrainingSet")
experiment_folder = os.path.join(root_folder,"ExampleExperiment")

########################################
# First we will train a model

# set the training set folder/ZIP file with manually sorted training set
# training set is a zip file containing folders with classes of manually sorted images
training_set_zip = os.path.join(training_set_folder,"TrainingSet.zip")

# set the folder where the trained model will be saved (along with some additional data)
# the model name will be inferred from the structure of the training set - the class names will be in the model name
training_results_folder= os.path.join(training_set_folder,"TrainingResults")

# now start the training
ml_tools.train_model(training_set_zip,training_results_folder)

########################################
#Next we will classify images from an experiment

# set the folder/ZIP file with images you want to classify

input_image_zip=os.path.join(experiment_folder,"ExampleExperimentImages.zip")

# set the path to the ml model you want to use
# if you used the training set from above, the model name will be 
# ml_model_classes_DoubleCells_NiceCells_DeadCells_Debris_DefectedCells
model_name = "ml_model_classes_DoubleCells_NiceCells_DeadCells_Debris_DefectedCells"
model_path=os.path.join(training_results_folder,"Models",model_name,model_name+".pt")

#set the path where classification data will be saved
output_classification_data_path= os.path.join(experiment_folder,"classification_data.txt")

# now start the classification
ml_tools.classify_images(input_image_zip,model_path,output_classification_data_path)

########################################
#Finally we will sort images from the experiment according to their classes and save them to zip files

#Now read the classification data into dataframe df
df = pd.read_csv(output_classification_data_path, sep='\t')
#Next clean df of extra spaces and NaN values
df=df.dropna()
df.rename(columns=lambda x: x.strip() if isinstance(x, str) else x, inplace=True)
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

#set the folder where to save sorted images in zip files
output_folder=os.path.join(experiment_folder,"ClassifiedImages")

#sort the images from the experiment (by default it will save just up to 1500 random images per class)
ml_tools.sort_class_images_from_zip(input_image_zip, df, output_folder,n_img=1500)