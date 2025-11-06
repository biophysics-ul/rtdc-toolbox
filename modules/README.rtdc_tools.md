# rtdc_tools

This module provides some useful functions for working with rtdc files. 

For the usage of these functions, see `rtdc_tools_example_script.py` in the `scripts` folder.


### Functions

`extract_features_from_rtdc_to_tsv(rtdc_path,tsv_path)`

This function saves all scalar features from a rtdc file into a tsv file. The rtdc file can be a rtdc file reffering to a basin. 

`extract_images_from_rtdc_to_zip(rtdc_path,zip_path)`

This function saves images from a rtdc file into a zip file. The rtdc file can be a rtdc file reffering to a basin (e.g., rtdc produced with ChipStream).  

If the rtdc file contains event "contours", the function extracts only "contoured events". You can set optional parameter `extra_pixels` to set the number of extra pixels on the left/right side of each contoured event to save with the extracted image. 

By default, it extracts images of "contoured events" with subtracted backgrounds. Set optional parameter to `subtract=False` to extract non-subtracted images.

`add_class_data_to_rtdc(input_rtdc_path,df_classes)`

This function reads an existing rtdc file defined in `input_rtdc_path`, uses the data stored in `df_classes` to create a new feature named "userdef1" and saves it into a new rtdc file (with added "_with_classes" to the filename). `df_classes` has to be a pandas dataframe with only one column containing integer numbers describing classes. The new rtdc file will reffer to the original rtdc. You can use the new rtdc file normally with CytoPlot.


