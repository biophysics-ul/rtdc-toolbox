# -*- coding: utf-8 -*-
"""
Module Name: dc_tools.py
Description: useful functions for working with Deformability Cytometry (DC) data
Authors: Darin Lah & Jure Derganc
Institute: Institute of Biophysics, University of Ljubljana
Last modified: 2025-05-07
License: GPL 3.0

Notes:
some functions work with images and data stored in a RTDC format
other functions work with images and data stored in a ZIP archive
"""

import cv2 as cv 
import numpy as np
import dclab
import pandas as pd
import zipfile
from tqdm import tqdm
import os
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


IMG_EXT=".png" # the extension of the images in the zip file

def get_img_diff(img1, img2, background_subtraction_shift =128):
    img_subtracted=(img1.astype(np.int32) + background_subtraction_shift - img2).astype(np.uint8)      
    return img_subtracted


def extract_features_from_rtdc_to_tsv(rtdc_path,tsv_path):
# reads a rtdc file and saves all scalar features into a tsv file    
    with dclab.new_dataset(rtdc_path) as ds:
        ds = dclab.new_dataset(rtdc_path)
        features = ds.features
        print("All features: ",features)    
        scalar_features = ds.features_scalar
        print("Scalar features: ",scalar_features)
        print("Extraction started...")
        data = {f: np.asarray(ds[f]) for f in scalar_features}    
        df = pd.DataFrame(data)
        df.to_csv(tsv_path, sep="\t", index=False)
        print(f"Scalar features saved to {tsv_path}")
        
    
def extract_images_from_rtdc_to_zip(rtdc_path,zip_path,subtract=True,extra_pixels=20,img_index_to_break=1000000):
# extracts images from a rtdc file - image filenames will be of the form: frame-event_index.IMG_EXT
# the rtdc file has to contain images
# if rtdc contains event contours, it will extract only events, otherwise it will extract the whole images
# subtract flag: if the background image should be subtracted from the image
# extra_pixels: how many pixels to add to the left and right of the contour
# img_index_to_break: how many images to extract (for testing purposes) 
    with dclab.new_dataset(rtdc_path) as ds:
            
        frame_index_previous=0 #these two are used to handle multiple events in an image
        event_index=1     
            
        (img_h,img_w)=ds["image"][0].shape
        
        contour_flag = "contour" in ds.features
        if "image" not in ds.features:
            print("rtdc file does not contain images! I did nothing.")
            return

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED, allowZip64=True) as zipf:
            print(f"Extracting images from rtdc to zip, countours={contour_flag}.")
            for i in tqdm(range(len(ds))):
                
                img=ds["image"][i]                
                
                if contour_flag:
                    img_bg=ds["image_bg"][i]
                    if subtract:
                        img=get_img_diff(img, img_bg)
                    x_contour_points=ds["contour"][i][:,0]  
                    x_min=max(0,min(x_contour_points)-extra_pixels)
                    x_max=min(img_w,max(x_contour_points)+extra_pixels)  
                    img=img[:, x_min:x_max]
                
                frame_index=ds["frame"][i]            
                if frame_index==frame_index_previous:
                    event_index=event_index+1 
                else:
                    event_index=1
                    frame_index_previous=frame_index
                    
                _, img_buffer = cv.imencode(IMG_EXT, img)
                png_filename=f"{frame_index:06}-{event_index}.png"
                zipf.writestr(png_filename, img_buffer.tobytes())            
                
                if i>=img_index_to_break:break  
               
def add_class_data_to_rtdc(input_rtdc_path,df_classes):
    # df classes has to be one column pandas dataframe with classes denoted wiht integer values (1,2,3...)
    # the length of df should be equal to the lenght of rtdc
    foldername = os.path.dirname(input_rtdc_path)
    basefilename, extension = os.path.splitext(os.path.basename(input_rtdc_path))
    output_rtdc_path = os.path.join(foldername, basefilename + "_with_classes" + extension)
    class_array= df_classes.to_numpy().flatten()
    
    with (dclab.new_dataset(input_rtdc_path) as ds,
          dclab.RTDCWriter(output_rtdc_path,mode='reset') as hw):
        if len(ds)!=len(class_array):
            print("Class info length is not the same as the RTDC lengt")
        # `ds` is the basin
        # `hw` is the referrer
    
        # First of all, we have to copy the metadata from the input file
        # to the output file. If we forget to do this, then dclab will
        # not be able to open the output file.
        hw.store_metadata(ds.config.as_dict(pop_filtering=True))
    
        # Next, we can compute and write the new feature to the output file.
        hw.store_feature("userdef1", class_array)
    
        # Finally, we write the basin information to the output file.
        hw.store_basin(
            basin_name="class data",
            basin_type="file",
            basin_format="hdf5",
            basin_locs=[input_rtdc_path],
        )
    
    # You can now open the output file and verify that everything worked.
    with dclab.new_dataset(output_rtdc_path) as ds_out:
        assert "userdef1" in ds_out, "check that the feature we wrote is there"
        assert "image" in ds_out, "check that we can access basin features"
        print(f"New RTDC file saved with {len(ds_out)} events: {output_rtdc_path}")

def label_for(name):
    try:
        return dclab.dfn.get_feature_label(name)
    except Exception:
        return name

def downsample_mask(a, target=2000, seed=None):
    a = a.astype(bool).copy()              # ensure boolean, work on a copy
    one_idx = np.flatnonzero(a)
    if one_idx.size <= target:
        return a
    rng = np.random.default_rng(seed)
    keep = rng.choice(one_idx, size=target, replace=False)
    a[:] = False
    a[keep] = True
    return a
        
def classify_WBCs_Kaliman(ds,echo=True):
    """
    Classifies rtdc events according to    
    Kaliman et al., Biophysical Journal, 2025.
    "Automation and improvement of WBC mechanical profiling in deformability cytometry"
    returns array of the same length as ds with classes denoted as integers denoted in DC_EVENT_CLASSES
    if echo==true, it prints the results at every step of classification
    it also returns two figures depicting the resuts
    """    
    # Step 1 box filters thresholds
    AREA_um_MIN = 30.0       # µm²
    AREA_um_MAX = 100.0      # µm²
    DEFORM_MAX = 0.12        
    AREA_RATIO_MAX = 1.10     # porosity <10% -> area_ratio <= 1.1

    # Step 2b brightness p90 window (bg-corrected)
    BRIGHT_P90_MIN = -10.0
    BRIGHT_P90_MAX =  15.0

    # Step 2c deformation window (original contour)
    DEFORM_RAW_MIN = 0.06
    DEFORM_RAW_MAX = 0.26

    # Lymph cleaning
    TEX_CONTRAST_MAX = 250.0

    RND_SEED = 42
    
    DC_EVENT_CLASSES = {
        "unclasified"       :0,
        "lymphocytes"       :21,
        "neutrophils"       :22,
        "monocytes"         :23,
        "eosinophils"       :24,
        "step1 discarded"   :91, 
        "step2 discarded"   :92,
        "step4 discarded"   :94,
        "step6 discarded"   :96
        }
    
    N = len(ds)
    if echo: print(f"Total events: {N:,}")
    
    # ---- pull features from dataset  ----
    area_um       = ds["area_um"][:]
    area_um_raw   = ds["area_um_raw"][:]
    area_ratio    = ds["area_ratio"][:]
    deform        = ds["deform"][:]   
    deform_raw    = ds["deform_raw"][:]         # no deform_cvx available
    bright_p90    = ds["bright_perc_90"][:]     # background-corrected P90
    bright_avg    = ds["bright_avg"][:]         # bg-corrected average brightness
    tex_con_avg   = ds["tex_con_avg"][:]        # texture contrast (avg)
    tex_den_avg   = ds["tex_den_avg"][:]        # difference entropy (avg)
    tex_sva_avg   = ds["tex_sva_avg"][:]        # sum variance (avg)
    tex_idm_ptp   = ds["tex_idm_ptp"][:]        # inverse difference moment (ptp)
    tex_den_ptp   = ds["tex_den_ptp"][:]        # difference entropy (ptp)
    
    # ======================== STEP 1: Crude WBC selection ============================
    crude_wbcs = (
        (area_um >= AREA_um_MIN) &
        (area_um <= AREA_um_MAX) &
        (deform   <= DEFORM_MAX) &
        (area_ratio <= AREA_RATIO_MAX) )  
    step1_discarded = ~ crude_wbcs
    if echo: print(f"Step 1 kept: {crude_wbcs.sum()} (discarded: {step1_discarded.sum()})")
    
    # ======================== STEP 2: Remove RBC doublets and other non target cells ==================
    # step 2a GMM, step 2b brightness box filter, step 2c deformation box filter
    base2 = crude_wbcs  # input cells for Step 2
    base2a=base2 #
    x=tex_idm_ptp[base2a] # Texture inverse difference moment (ptp)
    y=tex_den_ptp[base2a] # Texture difference entropy (ptp)
    data=np.column_stack((x,y))
    gmm_step2a = GaussianMixture(n_components=2, random_state=RND_SEED).fit(data)
    gmm_step2a_predictions = gmm_step2a.predict(data)
    centers = gmm_step2a.means_
    # Choose RBC doublets as the component with the larger center length (x^2 + y^2)
    norm2 = np.sum(centers**2, axis=1)      # [x^2 + y^2] for each component
    rbc_cluster = int(np.argmax(norm2))
    wbc_cluster = 1 - rbc_cluster
    # Build full-length boolean mask
    rbc_doublets_step2a = np.zeros(N, dtype=bool)
    wbc_step2a = np.zeros(N, dtype=bool)
    idx = np.flatnonzero(base2a)             # same mask you used to build `data`
    rbc_doublets_step2a[idx] = (gmm_step2a_predictions == rbc_cluster)
    wbc_step2a[idx] = (gmm_step2a_predictions == wbc_cluster)
    if echo: print(f"Step 2a (GMM) kept: {wbc_step2a.sum()}")
    
    # Step 2b: Brightness p90 box filter
    base2b = wbc_step2a 
    wbc_step2b = base2b & (bright_p90 >= BRIGHT_P90_MIN) & (bright_p90 <= BRIGHT_P90_MAX) #90th Percentile of brightness (bgc)
    if echo: print(f"Step 2b (brightness p90) kept: {wbc_step2b.sum():,}")
    
    # Step 2c: Deformation raw box filter 
    base2c = wbc_step2b
    wbc_step2c = base2c & (deform_raw >= DEFORM_RAW_MIN) & (deform_raw <= DEFORM_RAW_MAX)
    if echo: print(f"Step 2c (deformation raw) kept: {wbc_step2c.sum():,})")
    
    # finalizing Step 2:
    wbcs=wbc_step2c
    step2_discarded= base2 & (~wbcs)
    if echo: print(f"Step 2 kept: {wbcs.sum():,} (discarded: {step2_discarded.sum()})")
    
    # =================== Step 3 Lymphos vs Granulos & Monos (KMeans on size) ========================
    base3 = wbcs # input cells for Step 3
    x=area_um[base3]
    y=area_um_raw[base3]
    data=np.column_stack((x,y))
    km_step3 = KMeans(n_clusters=2, random_state=RND_SEED)
    km_step3_predictions = km_step3.fit_predict(data)
    centers = km_step3.cluster_centers_
    crude_lymph_cluster = np.argmin(centers.mean(axis=1))      # smaller area cluster = lymph
    crude_lymphos = np.zeros(N, dtype=bool)
    idx = np.flatnonzero(base3)
    crude_lymphos[idx]=(km_step3_predictions == crude_lymph_cluster)
    granulos_monos =base3 & (~crude_lymphos)
    if echo: print(f"Step 3 Crude Lymphos : {crude_lymphos.sum():,}")
    if echo: print(f"Step 3 Granulos and Monos: {granulos_monos.sum():,}")
    
    # ========== Step 4: Clean lymphos via box filter and GMM on (bright_bc_avg, tex_con_avg) =============
    base4a=crude_lymphos   
    step4a_accepted  = base4a & (tex_con_avg <= TEX_CONTRAST_MAX)
    x=bright_avg[step4a_accepted]  # Brightness average [a.u.]
    y=tex_con_avg[step4a_accepted] #Texture contrast (avg)
    data=np.column_stack((x,y))
    gmm_step4b = GaussianMixture(n_components=2, random_state=RND_SEED).fit(data)
    gmm_step4b_predictions = gmm_step4b.predict(data)
    centers = gmm_step4b.means_
    clean_lymph_cluster = int(np.argmax(centers[:, 0]))  # max mean along brightness axis
    clean_lymphos = np.zeros(N, dtype=bool)       # keep = cluster with max brightness
    idx = np.flatnonzero(step4a_accepted)            # positions used for GMM
    clean_lymphos[idx] = (gmm_step4b_predictions == clean_lymph_cluster)
    lymphocytes=clean_lymphos
    step4_discarded=base4a & (~lymphocytes)
    if echo: print(f"Step 4: Clean lymphos : {lymphocytes.sum():,}")
    
    # ======== Step 5: Neutrophil, monocyte, and eosinophil clustering via 3D GMM (bright, den_avg, sva_avg) 
    base5=granulos_monos
    x=bright_avg[base5] # Brightness average [a.u.]
    y=tex_den_avg[base5] #Texture difference entropy (avg)
    z=tex_sva_avg[base5]
    data=np.column_stack((x,y,z))
    gmm_step5 = GaussianMixture(n_components=3, random_state=RND_SEED).fit(data)
    gmm_step5_predictions = gmm_step5.predict(data)
    counts = np.bincount(gmm_step5_predictions, minlength=3)
    neut_cluster = int(np.argmax(counts)) #Find the most abundant component (neutrophils)
    neut_mask = np.zeros(N, dtype=bool)       # keep = cluster with max brightness
    idx = np.flatnonzero(base5)            # positions used for GMM
    neut_mask[idx] = (gmm_step5_predictions == neut_cluster)
    neutrophils = neut_mask
    monos_eos = base5 & (~neutrophils)
    if echo: print(f"Step 5: Neutrophils : {neutrophils.sum():,}")
       
    # ======== Step 6: Monocyte and eosinophil clustering box filter (tex_den_avg) 
    base6 = monos_eos 
    eosinophils = base6 & (tex_con_avg > tex_con_avg[neutrophils].max())
    monocytes = base6 & (tex_con_avg < tex_con_avg[neutrophils].min())
    step6_discarded = base6 & (~monocytes) & (~eosinophils)
    if echo: print(f"Step 5: Monocytes : {monocytes.sum():,}")
    if echo: print(f"Step 5: Eosinophils : {eosinophils.sum():,}")
    

        
    # Build the canvas
    fig1, axs = plt.subplots(4, 3, figsize=(14, 16), constrained_layout=True)
    ax = axs.ravel()  # 12 slots; we'll use 11

    # ---------- Panel 1: Step 1 ----------
    ds_mask = downsample_mask(step1_discarded, target=50_000)
    a = ax[0]
    a.scatter(area_um[ds_mask], deform[ds_mask], s=4, c="gray", alpha=0.2, label="discarded")
    a.scatter(area_um[crude_wbcs], deform[crude_wbcs], s=4, c="red", alpha=0.5, label="WBCs")
    a.axvline(AREA_um_MIN, linestyle="--")
    a.axvline(AREA_um_MAX, linestyle="--")
    a.axhline(DEFORM_MAX, linestyle="--")
    a.set_xlabel(label_for("area [µm²]"))
    a.set_ylabel(label_for("deform"))
    a.set_title(f"Step 1: Box filters (kept {crude_wbcs.sum():,} of total {N:,})")
    a.legend(fontsize=8)

    # ---------- Panel 2: Step 2a ----------
    a = ax[1]
    a.scatter(tex_idm_ptp[rbc_doublets_step2a], tex_den_ptp[rbc_doublets_step2a], s=6, alpha=0.5, c="gray", label="RBC doublets")
    a.scatter(tex_idm_ptp[wbc_step2a],        tex_den_ptp[wbc_step2a],        s=6, alpha=0.7, c="red",  label="WBCs")
    a.set_xlabel("tex_idm_ptp")
    a.set_ylabel("tex_den_ptp")
    a.set_title(f"Step 2a GMM clustering (kept: {wbc_step2a.sum():,})")
    a.legend(fontsize=8)

    # ---------- Panel 3: Step 2b ----------
    a = ax[2]
    vals = bright_p90[base2b]
    a.hist(vals, bins=64, alpha=0.8)
    a.axvline(BRIGHT_P90_MIN, linestyle="--")
    a.axvline(BRIGHT_P90_MAX, linestyle="--")
    a.set_xlabel("Brightness P90 (bg-corrected)")
    a.set_ylabel("Count")
    a.set_title(f"Step 2b (kept: {wbc_step2b.sum():,} of {base2b.sum():,})")

    # ---------- Panel 4: Step 2c ----------
    a = ax[3]
    vals = deform_raw[base2c]
    a.hist(vals, bins=64, alpha=0.8)
    a.axvline(DEFORM_RAW_MIN, linestyle="--")
    a.axvline(DEFORM_RAW_MAX, linestyle="--")
    a.set_xlabel(label_for("deform_raw"))
    a.set_ylabel("Count")
    a.set_title(f"Step 2c  (kept: {wbc_step2c.sum():,} of {base2c.sum():,})")

    # ---------- Panel 5: Step 2 final ----------
    a = ax[4]
    a.scatter(area_um[wbcs],            deform[wbcs],            s=4, c="red",  alpha=0.5, label="WBCs")
    a.scatter(area_um[step2_discarded], deform[step2_discarded], s=4, c="gray", alpha=0.2, label="RBC doublets")
    a.set_xlabel(label_for("area [µm²]"))
    a.set_ylabel(label_for("deform"))
    a.set_title(f"Step 2: Cleaned WBCs  (kept {wbcs.sum():,}/{step2_discarded.sum():,})")
    a.legend(fontsize=8)

    # ---------- Panel 6: Step 3 ----------
    a = ax[5]
    a.scatter(area_um[crude_lymphos], deform[crude_lymphos], s=4, c="red",  alpha=0.5, label="Lymphos")
    a.scatter(area_um[granulos_monos], deform[granulos_monos], s=4, c="gray", alpha=0.2, label="Granulos & Monos")
    a.set_xlabel(label_for("area [µm²]"))
    a.set_ylabel(label_for("deform"))
    a.set_title(f"Step 3: Crude Lymphos vs Granulos & Monos ({crude_lymphos.sum():,}/{granulos_monos.sum():,})")
    a.legend(fontsize=8)

    # ---------- Panel 7: Step 4 GMM ----------
    a = ax[6]
    a.scatter(bright_avg[lymphocytes],      tex_con_avg[lymphocytes],      s=4, c="red",  alpha=0.5, label="Lymphocytes")
    a.scatter(bright_avg[step4_discarded],  tex_con_avg[step4_discarded],  s=4, c="gray", alpha=0.5, label="Discarded")
    a.set_xlabel(label_for("bright_bc_avg"))
    a.set_ylabel(label_for("tex_con_avg"))
    a.set_title(f"Step 4: Clean Lymphocytes (kept {lymphocytes.sum():,} of {base4a.sum():,})")
    a.legend(fontsize=8)

    # ---------- Panel 8: Step 4 final ----------
    a = ax[7]
    a.scatter(area_um[lymphocytes],     deform[lymphocytes],     s=4, c="red",  alpha=0.5, label="Lymphocytes")
    a.scatter(area_um[step4_discarded], deform[step4_discarded], s=4, c="gray", alpha=0.5, label="Discarded")
    a.set_xlabel(label_for("area_um"))
    a.set_ylabel(label_for("deform"))
    a.set_title(f"Step 4: Clean Lymphocytes (kept {lymphocytes.sum():,} of {base4a.sum():,})")
    a.legend(fontsize=8)

    # ---------- Panel 9: Step 5 ----------
    a = ax[8]
    a.scatter(bright_avg[neutrophils], tex_den_avg[neutrophils], s=4, c="green", alpha=0.5, label="Neutrophils")
    a.scatter(bright_avg[monos_eos],   tex_den_avg[monos_eos],   s=4, c="gray",  alpha=0.5, label="Monos & Eosinos")
    a.set_xlabel(label_for("bright_avg"))
    a.set_ylabel(label_for("tex_den_avg"))
    a.set_title(f"Step 5: Neutrophils ({neutrophils.sum():,}/{monos_eos.sum():,})")
    a.legend(fontsize=8)

    # ---------- Panel 10: Step 6 ----------
    a = ax[9]
    a.scatter(area_um[neutrophils], tex_den_avg[neutrophils], s=4, c="green",  alpha=0.5, label=f"Neutrophils {neutrophils.sum()}")
    a.scatter(area_um[monocytes],   tex_den_avg[monocytes],   s=4, c="blue",   alpha=0.5, label=f"Monocytes {monocytes.sum()}")
    a.scatter(area_um[eosinophils], tex_den_avg[eosinophils], s=4, c="red",    alpha=0.5, label=f"Eosinophils {eosinophils.sum()}")
    a.set_xlabel(label_for("area_um"))
    a.set_ylabel(label_for("tex_den_avg"))
    a.set_title(f"Step 6 "
                f"({neutrophils.sum():,}/{monocytes.sum():,}/{eosinophils.sum():,})")
    a.legend(fontsize=8)

    # ---------- Panel 11: Final combined ----------
    a = ax[10]
    a.scatter(area_um[lymphocytes],  deform[lymphocytes],  s=4, c="blue",   alpha=0.5, label=f"Lymphocytes {lymphocytes.sum()}")
    a.scatter(area_um[neutrophils],  deform[neutrophils],  s=4, c="black",  alpha=0.5, label=f"Neutrophils {neutrophils.sum()}")
    a.scatter(area_um[monocytes],    deform[monocytes],    s=4, c="orange", alpha=0.5, label=f"Monocytes {monocytes.sum()}")
    a.scatter(area_um[eosinophils],  deform[eosinophils],  s=4, c="cyan",   alpha=0.5, label=f"Eosinophils {eosinophils.sum()}")
    a.set_xlabel(label_for("area_um"))
    a.set_ylabel(label_for("deform"))
    a.set_xlim(0, 130)
    a.set_ylim(0, 0.15)
    a.legend(fontsize=8)

    # ---------- Panel 12 (unused): turn off ----------
    a = ax[11]
    a.scatter(area_um[lymphocytes],  bright_avg[lymphocytes],  s=4, c="blue",   alpha=0.5, label=f"Lymphocytes {lymphocytes.sum()}")
    a.scatter(area_um[neutrophils],  bright_avg[neutrophils],  s=4, c="black",  alpha=0.5, label=f"Neutrophils {neutrophils.sum()}")
    a.scatter(area_um[monocytes],    bright_avg[monocytes],    s=4, c="orange", alpha=0.5, label=f"Monocytes {monocytes.sum()}")
    a.scatter(area_um[eosinophils],  bright_avg[eosinophils],  s=4, c="cyan",   alpha=0.5, label=f"Eosinophils {eosinophils.sum()}")
    a.set_xlabel(label_for("area_um"))
    a.set_ylabel(label_for("bright_avg"))
    a.set_xlim(0, 130)
    a.set_ylim(50, 200)
    a.legend(fontsize=8)
    

    fig1.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)
    plt.close(fig1)
        
    ###############################################################
        
    # Common mask for not classified events used in two panels
    mask = ~(lymphocytes & neutrophils & monocytes & eosinophils)
    downsampled_mask = downsample_mask(mask, target=10_000)
    
    # Build one figure with a 2×2 grid
    fig2, axs = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    
    # --- [0,0] Time vs Brightness (KDE-colored) ---
    xsamp, ysamp = ds.get_downsampled_scatter(xax="time", yax="bright_avg", downsample=10_000)
    kde = ds.get_kde_scatter(xax="time", yax="bright_avg", positions=(xsamp, ysamp))
    ax = axs[0, 0]
    ax.scatter(xsamp, ysamp, s=2, c=kde)  # add cmap/norm if you like
    ax.set_xlabel(dclab.dfn.get_feature_label("time"))
    ax.set_ylabel(dclab.dfn.get_feature_label("bright_avg"))
    
    # --- [0,1] Time vs Area (KDE-colored) ---
    xsamp, ysamp = ds.get_downsampled_scatter(xax="time", yax="area_um", downsample=50_000)
    kde = ds.get_kde_scatter(xax="time", yax="area_um", positions=(xsamp, ysamp))
    ax = axs[0, 1]
    ax.scatter(xsamp, ysamp, s=2, c=kde)
    ax.set_xlabel(dclab.dfn.get_feature_label("time"))
    ax.set_ylabel(dclab.dfn.get_feature_label("area_um"))
    
    
    # --- [1,0] Area vs Deform (classes) ---
    ax = axs[1, 0]
    ax.scatter(area_um[downsampled_mask], deform[downsampled_mask], s=4, c="grey",  alpha=0.5, label=f"Discarded ({mask.sum()})")
    ax.scatter(area_um[lymphocytes],      deform[lymphocytes],      s=4, c="blue",  alpha=0.5, label=f"Lymphocytes ({lymphocytes.sum()})")
    ax.scatter(area_um[neutrophils],      deform[neutrophils],      s=4, c="black", alpha=0.5, label=f"Neutrophils ({neutrophils.sum()})")
    ax.scatter(area_um[monocytes],        deform[monocytes],        s=4, c="orange",alpha=0.5, label=f"Monocytes ({monocytes.sum()})")
    ax.scatter(area_um[eosinophils],      deform[eosinophils],      s=4, c="cyan",  alpha=0.5, label=f"Eosinophils ({eosinophils.sum()})")
    ax.set_xlabel(label_for("area_um"))
    ax.set_ylabel(label_for("deform"))
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 0.7)
    ax.legend(fontsize=8, ncols=2, frameon=False)
    
    # --- [1,1] Area vs Brightness (classes) ---
    ax = axs[1, 1]
    ax.scatter(area_um[downsampled_mask], bright_avg[downsampled_mask], s=4, c="grey",  alpha=0.5, label=f"Discarded ({mask.sum()})")
    ax.scatter(area_um[lymphocytes],      bright_avg[lymphocytes],      s=4, c="blue",  alpha=0.5, label=f"Lymphocytes ({lymphocytes.sum()})")
    ax.scatter(area_um[neutrophils],      bright_avg[neutrophils],      s=4, c="black", alpha=0.5, label=f"Neutrophils ({neutrophils.sum()})")
    ax.scatter(area_um[monocytes],        bright_avg[monocytes],        s=4, c="orange",alpha=0.5, label=f"Monocytes ({monocytes.sum()})")
    ax.scatter(area_um[eosinophils],      bright_avg[eosinophils],      s=4, c="cyan",  alpha=0.5, label=f"Eosinophils ({eosinophils.sum()})")
    ax.set_xlabel(label_for("area_um"))
    ax.set_ylabel(label_for("bright_avg"))
    ax.set_xlim(0, 130)
    ax.set_ylim(50, 200)
    ax.legend(fontsize=8, ncols=2, frameon=False)
    plt.close(fig2)    

    userdef0 = np.zeros(N, dtype=np.int32)
    userdef0[lymphocytes] = DC_EVENT_CLASSES["lymphocytes"]
    userdef0[neutrophils] = DC_EVENT_CLASSES["neutrophils"]
    userdef0[monocytes]   = DC_EVENT_CLASSES["monocytes"]
    userdef0[eosinophils] = DC_EVENT_CLASSES["eosinophils"]
    userdef0[step1_discarded] = DC_EVENT_CLASSES["step1 discarded"]
    userdef0[step2_discarded] = DC_EVENT_CLASSES["step2 discarded"]
    userdef0[step4_discarded] = DC_EVENT_CLASSES["step4 discarded"]
    userdef0[step6_discarded] = DC_EVENT_CLASSES["step6 discarded"]   
   
    return userdef0, DC_EVENT_CLASSES, [fig1, fig2]
      
       