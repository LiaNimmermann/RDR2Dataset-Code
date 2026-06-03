import cv2
import numpy as np
import os
import tqdm

path_to_set = r"/media/lnimmermann/T5 EVO/RDR2_dataset_processed"
######## Loader and Helper functions ########

def load_image(file_path):
    if file_path.split(".")[-1].lower() == "exr":
        img = cv2.imread(file_path, flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR)
        #img = np.clip(img, 0, 10)
        # appy Reinhard-tone mapping
        #img /= (1.0 + img)
        #img *= 255
        img = img / (img.max() + 1e-8)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    else:
        img = cv2.imread(file_path)
    if img is None:
        msg = f"Failed to load image: {file_path}"
        raise ValueError(msg)
    return img.astype(np.uint8)

def get_images_from_id(id, time=0):
    if id < 1000000:
        id = 1000000 + id
    filename_exr = path_to_set + r"/HDR_exr/h_" + str(id) + "_"+ str(time) +".exr"
    filename_png = path_to_set + r"/PNG/o_" + str(id) + "_"+ str(time) +".png"
    return load_image(filename_exr), load_image(filename_png)

def load_exr_image(file_path):
    img = cv2.imread(file_path, flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR)
    if img is None:
        msg = f"Failed to load image: {file_path}"
        raise ValueError(msg)
    return img.astype(np.float32)

def load_exr_from_id(id, time=0):
    if id < 1000000:
        id = 1000000 + id
    filename_exr = path_to_set + r"/HDR_exr/h_" + str(id) + "_"+ str(time) +".exr"
    
    return load_exr_image(filename_exr)

def save_exr_images_to_png(ids, path_for_saving):
    for id in tqdm(ids):
        for time in [0,7,12,17,20]:
            exr = load_exr_from_id(id, time)
            cv2.imwrite(path_for_saving + r"/exr_to_png_{id}_{time}.png", exr)

def grayscale(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.astype(np.uint8)

########## Metrics Helper functions ##########

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

def similarity(img1, img2):
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return cv2.matchTemplate(
        img1, img2, cv2.TM_CCOEFF_NORMED
    )[0][0]

def get_symm_kl_div(img1, img2):
    hist1 = np.histogram(img1, bins=256, range=(0, 255), density=True)[0]
    hist2 = np.histogram(img2, bins=256, range=(0, 255), density=True)[0]
    kl_div = np.sum(hist1 * np.log((hist1 + 1e-10) / (hist2 + 1e-10)))
    symm_kl_div = kl_div + np.sum(hist2 * np.log((hist2 + 1e-10) / (hist1 + 1e-10)))
    return symm_kl_div

def get_mean_ratio(img1, img2):
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)
    mean_ratio = mean1 / mean2 if mean2 != 0 else 0
    return mean_ratio

def mse(img1, img2):
    h, w, c = img1.shape
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff**2)
    mse_value = err / (float(h * w * c))
    return mse_value#, diff

def compare_all_times_with_metric_min(id, metric_func, times="all"):
    exrs = []
    if times == "all":
        times = [0,7,12,17,20]
    elif times == "no_night":
        times = [7,12,17,20]
    else:
        times = [0,7,12,17,20]
    for itime in times:
        exr = load_exr_from_id(id, itime)
        exrs.append(exr)
    
    metric_min = float('inf')
    
    for i in range(len(exrs)-1):
        for j in range(i+1, len(exrs)):
            metric = metric_func(exrs[i], exrs[j])
            if metric < metric_min:
                metric_min = metric
    return metric_min

def compare_all_times_with_metric_mean(id, metric_func, times="all"):
    exrs = []
    if times == "all":
        times = [0,7,12,17,20]
    elif times == "no_night":
        times = [7,12,17,20]
    else:
        times = [0,7,12,17,20]
    for itime in times:
        exr = load_exr_from_id(id, itime)
        exrs.append(exr)
    
    metric_sum = 0
    for i in range(len(exrs)-1):
        for j in range(i+1, len(exrs)):
            metric_sum += metric_func(exrs[i], exrs[j])
    return metric_sum / 10  # Normalize by number of comparisons (10 for 5 images)

def compare_all_times_with_metric_conv(id, metric_func, times="all"):
    exrs = []
    if times == "all":
        times = [0,7,12,17,20]
    elif times == "no_night":
        times = [7,12,17,20]
    else:
        times = [0,7,12,17,20]
    for itime in times:
        exr = load_exr_from_id(id, itime)
        exr_convolved = convolve_with_DoG(exr)
        exrs.append(exr_convolved)
    
    metric_sum = 0
    for i in range(len(exrs)-1):
        for j in range(i+1, len(exrs)):
            metric_sum += metric_func(exrs[i], exrs[j])
    return metric_sum / 10  # Normalize by number of comparisons (10 for 5 images)

def compare_exr_with_png(id, time, metric_func):
    exr, png = get_images_from_id(id, time)
    return metric_func(exr, png)

def compare_all_times_exr_with_png_mean(id, metric_func, processing_func_exr=None, processing_func_png=None):
    exrs = []
    pngs = []
    for itime in [0,7,12,17,20]:
        exr, png = get_images_from_id(id, itime)
        if processing_func_exr:
            exr = processing_func_exr(exr)
        if processing_func_png:
            png = processing_func_png(png)
        exrs.append(exr)
        pngs.append(png)
    
    metric_sum = 0
    for i in range(len(exrs)):
        metric_sum += metric_func(exrs[i], pngs[i])
    return metric_sum / len(exrs)  # Normalize by number of comparisons


def compare_all_times_exr_with_png_min(id, metric_func, processing_func_exr=None, processing_func_png=None):
    exrs = []
    pngs = []
    for itime in [0,7,12,17,20]:
        exr, png = get_images_from_id(id, itime)
        if processing_func_exr:
            exr = processing_func_exr(exr)
        if processing_func_png:
            png = processing_func_png(png)
        exrs.append(exr)
        pngs.append(png)
    
    metric_min = float('inf')
    
    for i in range(len(exrs)):
        metric = metric_func(exrs[i], pngs[i])
        if metric < metric_min:
            metric_min = metric
    return metric_min  # Return the minimum metric


