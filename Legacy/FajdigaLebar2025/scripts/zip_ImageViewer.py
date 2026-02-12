# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 13:19:07 2025

@author: jderganc
"""

import numpy as np
import cv2 as cv2 # shape detection functions
import os # operating system functions
import zipfile
import tkinter as tk
from tkinter import filedialog
import sys

# Open a file dialog to select an RTDC file
root = tk.Tk()
root.withdraw()  # Hide the root window

input_ZIP_path = filedialog.askopenfilename(title="Select ZIP File", filetypes=[("ZIP File", "*.zip")])
if not input_ZIP_path:
    print("No file selected. Exiting...")
    sys.exit()

zip_filename = os.path.basename(input_ZIP_path)

n_cols=5
n_rows=7
font_scale = 0.5  # Small font size for text
font_thickness = 1  # Font thickness
border_size=1

def read_and_concatenate(zip_ref, file_list, start_idx, n_cols, n_rows):
    
    if start_idx + (n_cols * n_rows) > len(file_list):
        print("Not enough images in the list to form the grid.")
        return None
    
    images = []    
    max_width=0
    max_height=0
    
    for i in range(start_idx, start_idx + (n_cols * n_rows)):
        with zip_ref.open(file_list[i]) as image_file:
            img_array = np.frombuffer(image_file.read(), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)            
            if img is None:
                print(f"Error reading {file_list[i]}")
                return None
            images.append(img)
            height, width = img.shape
            max_width=max(max_width,width)
            max_height=max(max_height,height)
             
    final_image_w = (max_width+2*border_size)*n_cols
    final_image_h = (max_height+2*border_size)*n_rows
    final_image = np.zeros((final_image_h,final_image_w), dtype=np.uint8)
    
    for i in range(len(images)):
        height, width = images[i].shape
        row=i // n_cols
        col=i % n_cols
        #print(file_list[i]," ",images[i].shape," ",i," ",row," ",col)
        x=border_size+col*(2*border_size+max_width)
        y=border_size+row*(2*border_size+max_height)
        final_image[y:y+height,x:x+width]=images[i]
        cv2.putText(final_image, file_list[start_idx+i], (x,y+15), cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    return final_image
        

window_name=zip_filename
# Get screen size using Tkinter
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()


with zipfile.ZipFile(input_ZIP_path, 'r') as zip_ref:
    # Get a list of all files in the ZIP
    file_list = [f for f in zip_ref.namelist() if f.endswith(".png") or f.endswith(".tif")]    
    #print("List of images:", file_list)  # Print list of images
    start_idx=0
    max_idx=int(len(file_list)/(n_cols*n_rows))
    # Initialize OpenCV Window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    #cv2.setWindowProperty("Concatenated Images", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # Set window size slightly smaller than full screen (10px border)
    cv2.resizeWindow(window_name, screen_width - 40, screen_height - 100)
    cv2.moveWindow(window_name, 10, 10)
    
    # Callback function for trackbar
    def update_display(start_idx):
        start_idx = max(0, start_idx)  # Ensure valid index
        concatenated_image = read_and_concatenate(zip_ref, file_list, start_idx*n_cols*n_rows, n_cols, n_rows)
        if concatenated_image is not None:
            cv2.imshow(window_name, concatenated_image)
    
    # Create trackbar
    cv2.createTrackbar("Start Index", window_name, 0, max_idx, update_display)
    
    # Initial display
    update_display(0)
    
    # Wait for user interaction
    while True:
        key = cv2.waitKey(1)  # Check for key press every 1ms
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            key = 27  # Simulate pressing ESC
        
        if key == 27:  # ESC key
            break
    cv2.destroyAllWindows()