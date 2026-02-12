# -*- coding: utf-8 -*-
"""
This script scans a folder for rtdc files ending with model_name 
(e.g. `_cnt.rtdc` or `_klmn25.rtdc`),
analyzes them and saves the results as PNG images with the same base name
"""

from rtdc_toolbox import analyze_rtdc  
from pathlib import Path

def main():
    rtdc_folder = r"Z:\NaiadBackUp\KK\EDTA"
    model_name = "klmn25"
    overwrite = False  # analyze all files or only those without PNG
    class_feature = "userdef1"   # set to None to plot "All events"
    features_to_plot = ['area_um', 'deform','bright_avg','area_ratio']
    n_window = 10000 #window width
    step = 1000 #step for moving the window


    # first we'll identify the rtdc files to process
    p = Path(rtdc_folder)
    if not p.exists():
        raise FileNotFoundError(f"Folder not found: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"Not a folder: {p}")

    pattern = f"*_{model_name}.rtdc"
    all_rtdc_files = sorted(p.glob(pattern))

    if overwrite:
        rtdc_list = all_rtdc_files
    else:
        not_yet_analyzed = []
        for f in all_rtdc_files:
            # expected output PNG: "<basename>_<model_name>.png"
            out_png = f.with_name(f.stem + ".png")  # because f.stem already ends with _<model_name>
            if not out_png.exists():
                not_yet_analyzed.append(f)
        rtdc_list = not_yet_analyzed

    print("Starting rtdc analysis in folder: ",rtdc_folder, flush=True)

    for i, rtdc_path in enumerate(rtdc_list, start=1):
        print(f"Analysis started ({i}/{len(rtdc_list)}): {rtdc_path.name}", flush=True)

        out_png = analyze_rtdc(
            path_to_rtdc=rtdc_path,
            class_feature=class_feature,
            features_to_plot=features_to_plot,
            n_window=n_window,
            step=step,
        )

        print(f"Analysis finished: {out_png.name}", flush=True)


if __name__ == "__main__":
    main()
