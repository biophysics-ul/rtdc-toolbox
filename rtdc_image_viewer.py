# -*- coding: utf-8 -*-
"""
Version 2025 11 25
@author: Jure Derganc
Simple RTDC image viewer.
The images are showed in a N_COLS x N_ROWS grid
Press Space to toggle showing the contours (if they are available).
Pressing 0..9 shows only specified class (the class is encoded in CLASS_FEATURE)
The classes in CLASS_FEAURE should be encoded as integer values (0,1,2,...)
For each image we show:
    <event index - class> (top left) - the class is shown only if CLASS_FEATURE is present
    <event time> (top right)
    <confidence> (bottom left) - only if present
"""

import cv2 as cv2 # used because it has a build-in window/slider
import dclab
import tkinter as tk
from tkinter import filedialog
import os
import sys
import numpy as np
import rtdc_toolbox

# constants that define the appearence
N_COLS=5 
N_ROWS=10
FONT_SCALE = 0.5  
FONT_THICKNESS = 1  
BORDER_SIZE=10
TEXT_POSITION = (BORDER_SIZE , BORDER_SIZE + 10)

# constants that define where to look for class info
CLASS_FEATURE="userdef1"
CONFIDENCE_FEATURE="userdef2" 

#########################################################
def read_and_concatenate(ds,img_table:list,start_idx, n_cols, n_rows,img_width,img_height,border_size):
    # returns final_image which is a n_cols x n_rows grid of annotated images listed in img_table starting with start_idx
    final_image_w = (img_width+2*border_size)*n_cols
    final_image_h = (img_height+2*border_size)*n_rows
    n_img = min(n_cols*n_rows,len(img_table)-start_idx) # n images in collage        
    # first we'll make the large final image
    if plot_countours:
        final_image = np.zeros((final_image_h,final_image_w,3), dtype=np.uint8)
    else:   
        final_image = np.zeros((final_image_h,final_image_w), dtype=np.uint8)
    # next we'll position event images in the large final image    
    for i in range(n_img):             
       row=i // n_cols
       col=i % n_cols
       x=border_size+col*(2*border_size+img_width)
       y=border_size+row*(2*border_size+img_height)      
       img=ds["image"][img_table[start_idx+i]].copy()
       img_main_text=str(img_table[start_idx+i])       
       if CLASS_FEATURE in ds: 
            class_id = ds[CLASS_FEATURE][img_table[start_idx+i]]
            class_name = class_name_dict.get(class_id, str(class_id))
            img_main_text = img_main_text + "-" + class_name
       cv2.putText(img, img_main_text, TEXT_POSITION, cv2.FONT_HERSHEY_SIMPLEX,FONT_SCALE, 
                   (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)
       if CONFIDENCE_FEATURE in ds:
           img_confidence_text=str(ds[CONFIDENCE_FEATURE][img_table[start_idx+i]])
           cv2.putText(img, img_confidence_text, (BORDER_SIZE, img_height-10), 
                       cv2.FONT_HERSHEY_SIMPLEX,FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA) 
       if 'time' in ds:
           time_text=str(round(ds["time"][img_table[start_idx+i]],1))
           cv2.putText(img,time_text , (img_width-BORDER_SIZE-10-10*len(time_text),
                       BORDER_SIZE + 10 ), cv2.FONT_HERSHEY_SIMPLEX, 
                       FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA) 
       if plot_countours:
           img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
           img[ds["contour"][img_table[i+start_idx]][:,1], ds["contour"][img_table[i+start_idx]][:,0]] = [0, 0, 255] # Red in BGR
       
       final_image[y:y+img_height,x:x+img_width]=img               
    return final_image    

def filter_events(ds,event_class:int)->list:
    ''' sets a mask for events defined in event_class '''
    if event_class:
        mask = (ds[CLASS_FEATURE][:]==event_class)
        img_table = np.where(mask)[0].tolist()
    else:
        img_table = list(range(len(ds)))
    return img_table     
    
def _set_window_title(ds):
    """Update the OS window title with the current filter/class state."""
    title = f"RTDC: {rtdc_filename} |"
    if "contour" in ds:
        title += f" Contours: {'ON' if plot_countours else 'OFF'} (press space to toggle) |"
    else:
        title += " Contours not present |"
    if CLASS_FEATURE in ds:
        title +=  " Class filter: " + ("None" if active_class is None else f" {class_name_dict[active_class]} ")    
        if 0 in class_list:
           title += f" (press 0-{len(class_list-1)} to select class, Backspace/Delete to reset) |"
        else:
           title += f" (press 1-{len(class_list)} to select class, Backspace/Delete reset) |"
    else:
        title += " Class info not present |"
    title += " Press Esc to exit"
    # OpenCV 4.5+ provides setWindowTitle; keep running gracefully if unavailable.
    if hasattr(cv2, "setWindowTitle"):
        try:
            cv2.setWindowTitle(window_name, title)
        except cv2.error:
            pass
 
# Callback function for slider - called when slider is moved
def _update_display(start_idx:int):
    concatenated_image = read_and_concatenate(ds,img_table,start_idx*N_COLS*N_ROWS, N_COLS, N_ROWS,img_width,img_height,BORDER_SIZE)
    if concatenated_image is not None:
        cv2.imshow(window_name, concatenated_image)          
        
# For correct exit from the script after closing the window
def _window_is_open(name: str) -> bool:
    try:
        prop = cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE)
        # Closed / hidden / non-existent → treat as not open
        if prop <= 0:
            return False
    except cv2.error:
        # Some Qt builds throw instead of returning a value
        return False
    return True        
#########################################################        
#########################################################
# Main script starts here

# Open a file dialog to select an RTDC file
root = tk.Tk()
root.withdraw()  # Hide the root window
rtdc_path = filedialog.askopenfilename(title="Select RTDC File", filetypes=[("RTDC File", "*.rtdc")])
if not rtdc_path:
    print("No file selected. Exiting...")
    sys.exit()

# get class names if present
class_name_dict=rtdc_toolbox.get_classes_from_rtdc(rtdc_path)

with dclab.new_dataset(rtdc_path) as ds:
    print("Number of images in RTDC: ",len(ds))
    if not "image" in ds:
        print("Error. No images in rtdc ",rtdc_path)
        sys.exit()   
    img_height, img_width =ds["image"][0].shape    
    if CLASS_FEATURE in ds: 
        class_list=np.unique(ds[CLASS_FEATURE][:])
        if 0 in class_list:
            class_range=range(0, len(class_list))
            print("Class unclassified is present! Press 0,1,... to select class.")
        else:
            class_range=range(1, len(class_list) + 1)
            print("Press 1,... to select class.")
    else:
        print("Feature ", CLASS_FEATURE, " is not present in rtdc.")
        class_list=[]    
    
    plot_countours=False
    rtdc_filename = os.path.basename(rtdc_path)
    window_name = "RTDC Viewer"
    
    active_class = None  # None = all events, otherwise an int class label    
    img_table=filter_events(ds,active_class)
    start_idx=0
    max_idx=max(1,int(len(img_table)/(N_COLS*N_ROWS)))     
    
    # Create the main window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Image slider", window_name, 0, max_idx, _update_display)
    
    #############################################################################
    # Do some OpenCV aerobics to set full screen window 
    _set_window_title(ds)
    _update_display(0)
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
    ############################################################################## 
    # The main loop
    while True:   
        key = cv2.waitKey(0)   
        if key == ord(' ') and "contour" in ds:
            plot_countours= not plot_countours
            _set_window_title(ds)
            cur_idx = cv2.getTrackbarPos("Image slider", window_name)
            _update_display(cur_idx)
        elif ((ord('0') <= key <= ord('9')) and (key - ord('0') in class_range)) or key in (8, 127):
            if key in (8,127):
              img_table=filter_events(ds,None) 
              active_class = None
            else:  
              if 0 in class_list:  
                  class_idx=class_list[key-ord('0')]  # class 0 is unclassified
              else:    
                  class_idx=class_list[key-ord('1')]  
              img_table=filter_events(ds,class_idx)
              active_class = class_idx
            start_idx=0
            max_idx=max(1,int(len(img_table)/(N_COLS*N_ROWS)))
            cv2.setTrackbarMax("Image slider", window_name, max_idx)
            cv2.setTrackbarPos("Image slider", window_name, max_idx)
            cv2.setTrackbarPos("Image slider", window_name, start_idx)
            _set_window_title(ds)
            _update_display(start_idx)   
        elif key == 27:  # ESC
            break        
        if not _window_is_open(window_name):
            break 
cv2.destroyAllWindows()
