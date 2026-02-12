# -*- coding: utf-8 -*-
"""
Created on Thu May 22 22:56:19 2025

@author: jderganc
"""

import rtdc_tools
import os
import pandas as pd
import numpy as np 

folder_name=r"D:\RTDC"

basin_rtdc_filename=r"2025-03-05_14.03_M001_Sara_1.rtdc"
analyzed_rtdc_filename=r"2025-03-05_14.03_M001_Sara_1_dcn.rtdc"
basin_rtdc_path=os.path.join(folder_name,basin_rtdc_filename)
analyzed_rtdc_path=os.path.join(folder_name,analyzed_rtdc_filename)

# first we will extract all the scalar features into a tsv file
tsv_path=os.path.join(folder_name,analyzed_rtdc_filename[:-5]+".tsv")
rtdc_tools.extract_features_from_rtdc_to_tsv(analyzed_rtdc_path,tsv_path)

# next we would extract the images into a zip file
zip_path=os.path.join(folder_name,analyzed_rtdc_filename[:-5]+".zip")
rtdc_tools.extract_images_from_rtdc_to_zip(analyzed_rtdc_path,zip_path,img_index_to_break=1000)

zip_path=os.path.join(folder_name,basin_rtdc_filename[:-5]+".zip")
rtdc_tools.extract_images_from_rtdc_to_zip(basin_rtdc_path,zip_path,img_index_to_break=1000)


# next you would normally infere image classes from the tsv data / zip images
# and store the class data as new column to the tsv file
# in this script, we'll just create a random class data

# read the tsv into a datafame
df = pd.read_csv(tsv_path, sep='\t')

# define the column name with class data
class_column_name="img_class"

if class_column_name in df.columns:
    # class names are changed into integer values
    classes_array,uniques =pd.factorize(df[class_column_name])
    classes_array += 1
    df_classes=pd.DataFrame(classes_array)
else:
    # it class_column_name does not exist, we will just make a new random one
    df_classes = pd.DataFrame({class_column_name: np.random.randint(1, 6, size=len(df))})

rtdc_tools.add_class_data_to_rtdc(analyzed_rtdc_path,df_classes)
# this function makes a new rtdc file with class info
# the new rtdc file takes input_rtdc_path as its basin