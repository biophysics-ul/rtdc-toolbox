"""
Classify WBCs in RT-DC datasets.

This script scans RTDC_FOLDER for files matching "*_dcn.rtdc" that are not yet classified,
classifies the events, and saves results as "*_dcn_Kaliman2025.rtdc" (using the original
basename as a basin dataset).

Class labels are stored in the "userdef0" feature as:
    0 = unclassified
    1 = lymphocytes
    2 = neutrophils
    3 = monocytes
    4 = eosinophils

Classification follows:
Kaliman et al., Biophysical Journal, 2026.
"Automation and improvement of WBC mechanical profiling in deformability cytometry"

If SHOW_GRAPHS, then the script saves also 200 dpi PNGs
1. experiment_dcn.png 
- this is a global check of the experiment and the result of classification
2. experiment_dcn_Kaliman2025.png
- this shows the result of every classification step, as described in the paper

If OVERWRITE, then all the files are re-classified and the results overwritten
Version: 2025-11-26
"""

import os
#os.environ["OMP_NUM_THREADS"] ="1"
import numpy as np
import dclab
#import matplotlib.pyplot as plt
import rtdc_tools


# ============================ CONFIG =========================================

SHOW_GRAPHS = True
RTDC_FOLDER  = r"D:\Naiad\KK"
OVERWRITE = True# if overwrite previous _class files

# =============================================================================
# =============================================================================
# =============================================================================
# ================= Main script ===============================================


rtdc_folder_all_files = os.listdir(RTDC_FOLDER)
rtdc_folder_datasets_with_dcn = [s for s in rtdc_folder_all_files if s.endswith("_dcn.rtdc")]
rtdc_folder_datasets_with_class = [s for s in rtdc_folder_all_files if s.endswith("_dcn_Kaliman2025.rtdc")]

if OVERWRITE:
    for file in rtdc_folder_datasets_with_class:
        os.remove(os.path.join(RTDC_FOLDER,file))
    rtdc_files_to_process=rtdc_folder_datasets_with_dcn
else:
    basins_with_dcn = {s[:-9] for s in rtdc_folder_datasets_with_dcn}
    basins_with_class = {s[:-15] for s in rtdc_folder_datasets_with_class}
    basins_to_process = sorted(list(basins_with_dcn-basins_with_class))    
    rtdc_files_to_process= [s+"_dcn.rtdc" for s in basins_to_process]    
 


for rtdc_file_to_process in rtdc_files_to_process:
    rtdc_in = os.path.join(RTDC_FOLDER,rtdc_file_to_process)
    base = os.path.splitext(rtdc_in)[0]
    rtdc_out  = os.path.join(base+"_Kaliman2025.rtdc")
    print("Loading:", rtdc_in)
    with dclab.new_dataset(rtdc_in) as ds, dclab.RTDCWriter(rtdc_out) as hw:
        # Copy essential metadata so the new file is a valid RT-DC dataset
        hw.store_metadata(ds.config.as_dict(pop_filtering=True))
        # Compute and store one scalar per event
        try: 
            userdef0,DC_EVENT_CLASSES,[fig1,fig2] = rtdc_tools.classify_WBCs_Kaliman(ds)
                        
            if SHOW_GRAPHS:
                fig2.savefig(base + ".png", dpi=200, bbox_inches="tight", facecolor="white")
                fig1.savefig(base+"__Kaliman2025.png", dpi=200, bbox_inches="tight", facecolor="white")    
            
            INVERSE_DC_EVENT_CLASSES = {v: k for k, v in DC_EVENT_CLASSES.items()}               
            print("\nFinal counts:")
            print(f"Total events: {len(ds)}")
            for dc_class in DC_EVENT_CLASSES.values():
                print(f"Class {dc_class} ({INVERSE_DC_EVENT_CLASSES[dc_class]}): {np.sum(userdef0==dc_class):,}")
        
            assert len(userdef0) == len(ds), "userdef0 must have one value per event"
            hw.store_feature("userdef0", userdef0)
            hw.store_basin(
                basin_name="raw data",
                basin_type="file",
                basin_format="hdf5",   # .rtdc container
                basin_locs=[rtdc_in],
                # basin_map=None       # omitted because lengths match
            )
        except:
            pass
    
    # (optional) sanity check
    with dclab.new_dataset(rtdc_out) as dso:
        try:
            assert "userdef0" in dso.features_innate
            assert len(dso) == len(dclab.new_dataset(rtdc_in))
            print("OK:", len(dso), "events; userdef0 present; basin linked.\n")
        except:
            print("No good.")