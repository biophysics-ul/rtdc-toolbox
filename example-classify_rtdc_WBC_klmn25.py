"""
This script scans a folder for rtdc files ending with rtdc_ext (e.g. `_dcn.rtdc`) 
The cells in the rtdc have to be segmented and their rtdc features calculated
and runs classify_rtdc_WBC_klmn25 on them
This classification follows 
    Kaliman et al., Biophysical Journal, 2025, DOI: 10.1016/j.bpj.2025.10.007
    "Automation and improvement of WBC mechanical profiling in deformability cytometry"
The result of classification is saved as a separate rtdc file with model_name in extension (e.g. `_dcn_klmn25.rtdc` )
"""

from rtdc_toolbox import classify_rtdc_WBC_klmn25
from pathlib import Path

def main():
    rtdc_folder=r"Z:\NaiadBackUp\KK\EDTA"
    rtdc_ext=r"_dcn.rtdc" #extension of rtdc files to be classified
    model_name="klmn25"
    overwrite=False # classify all files or only those which are not yet classified
    classified_rtdc_ext="_"+model_name
    
    
    # first we'll identify the rtdc files to process
    p = Path(rtdc_folder)
    if not p.exists():
       raise FileNotFoundError(f"Folder not found: {p}")
    if not p.is_dir():
       raise NotADirectoryError(f"Not a folder: {p}")

    all_rtdc_files = sorted(p.glob("*"+rtdc_ext))
    if overwrite:
        rtdc_list=all_rtdc_files
    else:
        not_yet_classified = []
        for f in all_rtdc_files:
            # expected output: replace suffix "_dcn.rtdc" -> "_dcn_klmn25.rtdc"
            classified_file = f.with_name(f.stem.replace(rtdc_ext[:-5], rtdc_ext[:-5] + classified_rtdc_ext) + ".rtdc")
            if not classified_file.exists():
                not_yet_classified.append(f)
        rtdc_list=not_yet_classified
             
    print("Starting classifying rtdc files in folder: ", rtdc_folder, flush=True)

    for i, rtdc_path in enumerate(rtdc_list, start=1):
        print(f"Classification started ({i}/{len(rtdc_list)}): {rtdc_path}", flush=True)
        classify_rtdc_WBC_klmn25(
            path_to_rtdc= str(rtdc_path),
            class_feature = "userdef1",
            echo=True,
            out_ext=classified_rtdc_ext
        )
    
        print("Classification finished:", rtdc_path, flush=True)


if __name__ == "__main__":
    main()
