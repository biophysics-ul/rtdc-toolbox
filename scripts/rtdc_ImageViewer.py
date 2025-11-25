# -*- coding: utf-8 -*-
"""
Version 2025 11 25
@author: Jure Derganc
Simple RTDC image viewer.
Press Space to toggle showing the contours (if they are available).
"""

import cv2 as cv2 # shape detection functions
import dclab
import tkinter as tk
from tkinter import filedialog
import os
import sys

# constants that define the appearence
N_COLS=5
N_ROWS=10
FONT_SCALE = 0.5  
FONT_THICKNESS = 1  
BORDER_SIZE=10
TEXT_POSITION = (BORDER_SIZE , BORDER_SIZE + 10)

# Open a file dialog to select an RTDC file
root = tk.Tk()
root.withdraw()  # Hide the root window
rtdc_path = filedialog.askopenfilename(title="Select RTDC File", filetypes=[("RTDC File", "*.rtdc")])
if not rtdc_path:
    print("No file selected. Exiting...")
    sys.exit()

plot_countours=False
rtdc_filename = os.path.basename(rtdc_path)
window_name="RTDC: "+rtdc_filename+" Press spacebar to toggle showing contours. Press ESC to exit."

ds = dclab.new_dataset(rtdc_path)
print("Number of images in RTDC: ",len(ds))

if not "image" in ds:
    print("No images in rtdc. Exiting...")
    sys.exit()

def read_and_concatenate(start_idx, n_cols, n_rows):

    images = []    
    
    if start_idx + (n_cols * n_rows) > len(ds):
        print("Not enough images in the list to form the grid.")
        return None
    
    for i in range(start_idx, start_idx + (n_cols * n_rows)):
        img=ds["image"][i]
        img_frame=ds["frame"][i] # index of the image
        cv2.putText(img, str(img_frame), TEXT_POSITION, cv2.FONT_HERSHEY_SIMPLEX, 
                    FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)
        if plot_countours:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img[ds["contour"][i][:,1], ds["contour"][i][:,0]] = [0, 0, 255] # Red in BGR
        
        images.append(cv2.copyMakeBorder(img, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE,
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    grid = []
    for i in range(n_rows):
        row = images[i * n_cols:(i + 1) * n_cols]  # Extract row
        row_concat = cv2.hconcat(row)  # Concatenate row images horizontally
        grid.append(row_concat)
    final_image = cv2.vconcat(grid)  # Concatenate all rows vertically
    return final_image
        

start_idx=0
max_idx=int(len(ds)/(N_COLS*N_ROWS))

# Create the main window
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Callback function for trackbar
def update_display(start_idx):
    start_idx = max(0, start_idx)  # Ensure valid index
    concatenated_image = read_and_concatenate(start_idx*N_COLS*N_ROWS, N_COLS, N_ROWS)
    if concatenated_image is not None:
        cv2.imshow(window_name, concatenated_image)

# Create trackbar
cv2.createTrackbar("Image slider", window_name, 1, max_idx, update_display)

# Do some OpenCV aerobics to set full screen window 
update_display(1)
cv2.waitKey(1)
cv2.setWindowProperty(window_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
_, _, screen_w, screen_h = cv2.getWindowImageRect(window_name)
cv2.setWindowProperty(window_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_NORMAL)
cv2.waitKey(1)
win_w = int(screen_w * 0.9)
win_h = int(screen_h * 0.9)
cv2.resizeWindow(window_name, win_w, win_h)
x = (screen_w - win_w) // 2
y = (screen_h - win_h) // 2
cv2.moveWindow(window_name, x, y)

# For save exiting after closing the window
def window_is_open(name: str) -> bool:
    try:
        prop = cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE)
        # Closed / hidden / non-existent → treat as not open
        if prop <= 0:
            return False
    except cv2.error:
        # Some Qt builds throw instead of returning a value
        return False
    return True

# The main loop
while True:

    key = cv2.waitKey(0)

    if key == ord(' ') and "contour" in ds:
        plot_countours= not plot_countours
        cur_idx = cv2.getTrackbarPos("Image slider", window_name)
        update_display(cur_idx)
        
    if key == 27:  # ESC
        break
    
    if not window_is_open(window_name):
        break
 
cv2.destroyAllWindows()