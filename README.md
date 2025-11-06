# rtdc-tools

Custom Python tools for Deformability Cytometry (DC).  
These tools are designed for reusability rather than speed.  
The repository contains two main modules in the folder `modules`and some useful scripts in the folder `scripts`.

### 1. Module `rtdc_tools.py`
This module provides functions for working with DC images. It supports images stored in the RTDC file format or in ZIP files. RTDC is the standard file format for deformability cytometry (see [**dclab**](https://github.com/DC-analysis/dclab)).

### 2. Module `ml_tools.py`
This module provides functions for simple ML-based model training and image classification.  
It is independent of DC (and `rtdc_tools.py`) and works with any images stored in ZIP files.  
Machine learning is based on PyTorch and can utilize a GPU. To use GPU acceleration, make sure CUDA is installed and properly configured.

### 3. Useful scripts
Some useful scripts and examples are in the folder `scripts`:
- `rtdc_ImageViewer`: Script for previewing images in RTDC files 
- `zip_ImageViewer`: Script for previewing images in ZIP files
- `ml_tools_example_script`: Script with a complete ML workflow example (model training & image classification)
- `rtdc_tools_example_script`: Script with examples of how to work with RTDC files (e.g., extract data, add class info ...)

## Usage

- Install the required packages (see below)
- Download the files, place the modules and scripts in the same folder on your computer, and you're ready to go!

Additional information about the modules is available in the module-specific README files:
- [README.rtdc_tools](/modules/README.rtdc_tools.md)
- [README.ml_tools](/modules/README.ml_tools.md)

Additional information about the scripts is included in the scripts themselves.



## Installation

- Create a new Python environment  
- Go to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/), select your system configuration, copy the installation command, and run it in your console  
- For our configuration, the correct command was:

```
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

- Then install the remaining packages:

```
    pip install numpy opencv-python dclab tqdm matplotlib scikit-learn seaborn
```

If you won't use `rtdc_tools.py`, you can skip installing `dclab`.  
If you won't use `ml_tools.py`, you can skip `torch` and `torchvision`.  

If installing current versions of packages doesn't work, try the versions that worked for us:
```
Python        3.13.2
numpy         2.2.6
opencv-python 4.11.0.86
dclab         0.64.0
tqdm          4.67.1
matplotlib    3.10.3
scikit-learn  1.6.1
seaborn       0.13.2
torch         2.7.0+cu126
torchvision   0.22.0+cu126
```

---

**Authors:**  
Darin Lah  
Bor Ivanuš  
Jure Derganc  
Institute of Biophysics, Faculty of Medicine, University of Ljubljana
