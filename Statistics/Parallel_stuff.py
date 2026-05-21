# load exr image in numpy array
import pickle

import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
from pathlib import Path
from tqdm import tqdm

def load_image(file_path):
    if file_path.split(".")[-1].lower() == "exr":
        img = cv2.imread(file_path, flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR)
        img = np.clip(img, 0, 10)
        # appy Reinhard-tone mapping
        img /= (1.0 + img)
        img *= 255
    else:
        img = cv2.imread(file_path)
    if img is None:
        msg = f"Failed to load image: {file_path}"
        raise ValueError(msg)
    return img.astype(np.uint8)


def get_images_from_id(id, time=0):
    if id < 1000000:
        id = 1000000 + id
    filename_exr = r"D:\Subset\HDR_EXR\h_" + str(id) + "_"+ str(time) +".exr"
    filename_png = r"D:\Subset\PNG\o_" + str(id) + "_"+ str(time) +".png"
    return load_image(filename_exr), load_image(filename_png)

def load_exr_image(file_path):
    img = cv2.imread(file_path, flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR)
    if img is None:
        msg = f"Failed to load image: {file_path}"
        raise ValueError(msg)
    return img.astype(np.float32)

def convolve_with_DoG(image, kernel_size=5, sigma=1.0):
    # Create a Difference of Gaussians (DoG) kernel
    k1 = cv2.getGaussianKernel(kernel_size, sigma)
    k2 = cv2.getGaussianKernel(kernel_size, sigma * 1.6)
    dog_kernel = k1 - k2

    normalized_dog_kernel = dog_kernel / np.sum(np.abs(dog_kernel))
    # Convolve the image with the DoG kernel
    convolved_image = cv2.filter2D(src= image, ddepth=-1, kernel=normalized_dog_kernel)

    # Sparsify the convolved image by setting values below a threshold to zero
    threshold = 0.2 * np.max(convolved_image)
    convolved_image[convolved_image < threshold] = 0
    convolved_image[convolved_image >= threshold] = 255
    
    
    return convolved_image

def load_exr_from_id(id, time=0):
    if id < 1000000:
        id = 1000000 + id
    filename_exr = r"D:\Subset\HDR_EXR\h_" + str(id) + "_"+ str(time) +".exr"
    
    return load_exr_image(filename_exr)

def get_symm_kl_div(img1, img2):
    hist1 = np.histogram(img1, bins=256, range=(0, 255), density=True)[0]
    hist2 = np.histogram(img2, bins=256, range=(0, 255), density=True)[0]
    kl_div = np.sum(hist1 * np.log((hist1 + 1e-10) / (hist2 + 1e-10)))
    symm_kl_div = kl_div + np.sum(hist2 * np.log((hist2 + 1e-10) / (hist1 + 1e-10)))
    return symm_kl_div

def mse(img1, img2):
    h, w, c = img1.shape
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff**2)
    mse_value = err / (float(h * w * c))
    return mse_value, diff

# Iterate over all files in the directory
import multiprocessing
import tqdm.auto

directory = r"D:\Subset\HDR_EXR"
divs = {}

def process_file(filename):
    id = int(filename.split("_")[1])
    time = filename.split("_")[2].split(".")[0]
    if time == "0":
        exrs = []
        #pngs = []
        div = 0
        for itime in [0,7,12,17,20]:
            exr = load_exr_from_id(id, itime)
            exr_convolved = convolve_with_DoG(exr)
            exrs.append(exr_convolved)
            #pngs.append(cv2.Canny(png, 100, 200))
        
        for i in range(len(exrs)-1):
            for j in range(i+1, len(exrs)):
                div += mse(exrs[i], exrs[j])[0]
        return id, div
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
    if False:   #mean ratio after canny
        exr_canny = cv2.Canny(exr, 100, 200)
        png_canny = cv2.Canny(png, 100, 200)
        mean_ratio_after_canny = np.mean(exr_canny/(png_canny + 1e-10))
        
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
    pickle.dump(results, open("conv_mse_per_id_02.pkl", "wb"))    
    print(results)