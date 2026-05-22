# load exr image in numpy array
import pickle

import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
from pathlib import Path
from tqdm import tqdm

from helper import *

# Iterate over all files in the directory
import multiprocessing
import tqdm.auto

directory = r"D:\Subset\HDR_EXR"
divs = {}

##### Function to process each file and compute the metric #####
def process_file(filename):
    id = int(filename.split("_")[1])
    time = filename.split("_")[2].split(".")[0]
    if time == "0":     # only process each ID once, using the first time step
        
        ##### TODO: Find useful metric to filter bad images to later filter per threshold #####
        metric = compare_all_times_with_metric(id, metric_func=mse)
        #metric = compare_all_times_with_metric(id, metric_func=get_mean_ratio)
        #metric = compare_all_times_with_metric(id, metric_func=get_symm_kl_div)
        #metric = compare_all_times_exr_with_png(id, metric_func=mse, processing_func_exr=convolve_with_DoG, processing_func_png=convolve_with_DoG)
        #metric = ...?
        return id, metric
    else:
        return 0,0
        
    
    id_time = (id, time)
    exr, png = get_images_from_id(id, int(time))
    # Save exr to png file
    cv2.imwrite(f"D:\\Subset\\EXR_to_PNG\\exr_to_png_{id}_{time}.png", exr)
    
    if False: #symm_kl_div
        hist_exr = np.histogram(exr, bins=256, range=(0, 255), density=True)[0]
        hist_png = np.histogram(png, bins=256, range=(0, 255), density=True)[0]
        kl_div = np.sum(hist_exr * np.log((hist_exr + 1e-10) / (hist_png + 1e-10)))
        symm_kl_div = kl_div + np.sum(hist_png * np.log((hist_png + 1e-10) / (hist_exr + 1e-10)))
    if False:   #mean ratio
        exr_mean = np.mean(exr)
        png_mean = np.mean(png)
        mean_ratio = exr_mean / png_mean if png_mean != 0 else 0
        
    return id_time, 0

if __name__ == '__main__':
    multiprocessing.freeze_support()
    with multiprocessing.Pool(os.cpu_count()) as process_pool:
        results = dict(
            tqdm.auto.tqdm(
                process_pool.imap(
                    process_file,
                    os.listdir(directory),
                ),
                total=len(os.listdir(directory)),
                desc="Doing stuff",
                unit="file",
            ),
        )
    results = {id: div for id, div in results.items() if id != 0}
    pickle.dump(results, open("id_metric_dict.pkl", "wb"))    
    print(results)
    print("-"*20)
    