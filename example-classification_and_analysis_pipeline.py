
from rtdc_toolbox import classify_rtdc_ml, preprocess_rtdc_image, classify_rtdc_WBC_klmn25, analyze_rtdc, export_examples_of_cells_for_classes_in_rtdc
import os
'''
A simple script for classification of a rtdc file segmented with ChipStream ( _dnc.rtdc)
If classifies with 
'''

def main():
    
    # this has to be called from the main function not from top-level (in the later case, you get some errors with multiprocessing)
    path_to_rtdc_folder=r"D:\RTDC\Sample"
    original_rtdc="Sample1_dcn.rtdc"
    path_to_model_folder=r"D:\RTDC\Training\Models"
    
    # first we'll classify with specified ML models
    ml_models=["convnext_tiny_Aggregates_Ghosts_Platelets_RBC_RBCDoublets_RBCLised_WBC.pt"]
    models_ext=["_cnt"]   

    print(f"ML Classification started: {original_rtdc} in folder {path_to_rtdc_folder}", flush=True)
    for i, model in enumerate(ml_models):
        print("Classification started for model ",model)
        classified_ml_rtdc_ext=models_ext[i]
        
        classify_rtdc_ml(
            path_to_rtdc=os.path.join(path_to_rtdc_folder,original_rtdc),
            path_to_model=os.path.join(path_to_model_folder,model),
            preprocessing_function=preprocess_rtdc_image,
            class_feature="userdef1",
            prob_feature="userdef2",
            extra_pixels=15,
            overwrite=True,
            show_progress=True,
            batch_size=2048,
            num_workers=6,
            prefetch_batches=8,
            model_ext=classified_ml_rtdc_ext
        )
    # next well classify with Kaliman 2025    
    print(f"Kaliman 2025 classification started: {original_rtdc} in folder {path_to_rtdc_folder}", flush=True)
    classify_rtdc_WBC_klmn25(
        path_to_rtdc=os.path.join(path_to_rtdc_folder,original_rtdc),
        class_feature="userdef1",
        echo=True,
        overwrite=True,
        out_ext="_klmn25"
    )    
    
    #next we'll analyze all the classified rtdc files    
    models_ext +=["_klmn25"] 
    features_to_plot = ['area_um', 'deform','bright_avg','area_ratio']
    n_window = 10000
    step = 1000

    for model in models_ext:
        print("Analysis started for model: ", model, flush=True)
        analyze_rtdc(path_to_rtdc=os.path.join(path_to_rtdc_folder,original_rtdc[:-5]+model+".rtdc"),
                     class_feature="userdef1",
                     features_to_plot=features_to_plot,
                     n_window=n_window,                     
                     step=step)  

        print("Exporting PNG of cell examples for model: ", model, flush=True)
        export_examples_of_cells_for_classes_in_rtdc(
            path_to_rtdc=os.path.join(path_to_rtdc_folder,original_rtdc[:-5]+model+".rtdc"),
            report_ext="_classes",                  # suffix added to basename before .png
            N_img=40,                            # max images per class (after filtering)
            class_feature="userdef1",            # RTDC feature containing class id
            prob_feature="userdef2",              # RTDC feature containing probability/confidence (optional)
            preprocessing_function=preprocess_rtdc_image,  # None -> default preprocess_rtdc_image
            randomize=True,                    # choose random N_img per class vs first N_img
            prob_thresh=0.95,                # threshold value (used only if prob_feature exists)
            keep_above_thresh=True,    # above vs below threshold selection
            n_rows=4,                          # how many rows per class grid
)

    
if __name__ == "__main__":
    # optional but harmless; helpful for some Windows setups
    import multiprocessing as mp
    mp.freeze_support()
    main()
