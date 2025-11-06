import cv2
import numpy as np
import os
from tqdm import tqdm

folder = r'C:\Users\dekav\VSC-Code\RDR2Dataset-Code\converted_exr_val'  # replace with your HDR folder
max_values = []

for filename in tqdm(os.listdir(folder), desc='Image files'):
    if filename.lower().endswith('.exr'):
        filepath = os.path.join(folder, filename)
        hdr = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)  # H x W x C, float32
        if hdr is not None:
            max_values.append(hdr.max())
        else:
            print(f"Warning: failed to read {filepath}")

max_values = np.array(max_values)
print("Max value stats:")
print("Mean:", max_values.mean())
print("Median:", np.median(max_values))
print("99th percentile:", np.percentile(max_values, 99))