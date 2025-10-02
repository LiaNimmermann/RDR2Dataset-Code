import imageio.v3 as iio
import os
import imagecodecs

import numpy as np
import OpenEXR, Imath


def save_exr_32(filename, img: np.ndarray):
    """
    Save a float32 NumPy array as 32-bit EXR.
    Supports grayscale (H,W), RGB (H,W,3), RGBA (H,W,4).
    """
    if img.dtype != np.float32:
        raise ValueError("EXR requires float32 data")

    H, W = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1

    # Pixel type
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)  # 32-bit float

    # Define header and channels
    header = OpenEXR.Header(W, H)
    if channels == 1:
        header['channels'] = {'R': Imath.Channel(FLOAT)}
        exr = OpenEXR.OutputFile(filename, header)
        exr.writePixels({'R': img.tobytes()})
    elif channels == 3:
        header['channels'] = {
            'R': Imath.Channel(FLOAT),
            'G': Imath.Channel(FLOAT),
            'B': Imath.Channel(FLOAT)
        }
        exr = OpenEXR.OutputFile(filename, header)
        exr.writePixels({
            'R': img[:, :, 0].astype(np.float32).tobytes(),
            'G': img[:, :, 1].astype(np.float32).tobytes(),
            'B': img[:, :, 2].astype(np.float32).tobytes()
        })
    elif channels == 4:
        header['channels'] = {
            'R': Imath.Channel(FLOAT),
            'G': Imath.Channel(FLOAT),
            'B': Imath.Channel(FLOAT),
            'A': Imath.Channel(FLOAT)
        }
        exr = OpenEXR.OutputFile(filename, header)
        exr.writePixels({
            'R': img[:, :, 0].astype(np.float32).tobytes(),
            'G': img[:, :, 1].astype(np.float32).tobytes(),
            'B': img[:, :, 2].astype(np.float32).tobytes(),
            'A': img[:, :, 3].astype(np.float32).tobytes()
        })
    else:
        raise ValueError("Unsupported number of channels")

    exr.close()

def save_exr(filename, img: np.ndarray):
    """
    Save a float32 NumPy array as 32-bit EXR.
    Supports grayscale (H,W), RGB (H,W,3), RGBA (H,W,4).
    """
    if img.dtype != np.float32:
        raise ValueError("EXR requires float32 data")
    
    H, W = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)  # 32-bit float

    # Set up header with 32-bit channels
    header = OpenEXR.Header(W, H)
    
    if channels == 1:
        header['channels'] = {'R': Imath.Channel(FLOAT)}
        exr = OpenEXR.OutputFile(filename, header)
        exr.writePixels({'R': img.tobytes()})
    
    elif channels == 3:
        header['channels'] = {
            'R': Imath.Channel(FLOAT),
            'G': Imath.Channel(FLOAT),
            'B': Imath.Channel(FLOAT)
        }
        exr = OpenEXR.OutputFile(filename, header)
        exr.writePixels({
            'R': img[:, :, 0].tobytes(),
            'G': img[:, :, 1].tobytes(),
            'B': img[:, :, 2].tobytes()
        })
    
    elif channels == 4:
        header['channels'] = {
            'R': Imath.Channel(FLOAT),
            'G': Imath.Channel(FLOAT),
            'B': Imath.Channel(FLOAT),
            'A': Imath.Channel(FLOAT)
        }
        exr = OpenEXR.OutputFile(filename, header)
        exr.writePixels({
            'R': img[:, :, 0].tobytes(),
            'G': img[:, :, 1].tobytes(),
            'B': img[:, :, 2].tobytes(),
            'A': img[:, :, 3].tobytes()
        })
    else:
        raise ValueError("Unsupported number of channels")
    
    exr.close()


def convert_batch(input_folder, output_folder):
    """
    Convert all JXR files in input_folder to EXR files in output_folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".jxr"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".exr")
            
            with open(input_path, "rb") as f:
                data = f.read()

            try:
                img = imagecodecs.jpegxr_decode(data)
                print("Decoded JXR shape:", img.shape, "dtype:", img.dtype)
                save_exr_32(output_path, img)
                print(f"Converted {filename} -> {output_path}")
            except Exception as e:
                print("Failed to decode JXR:", e)
        
            
convert_batch("D:/Test_dataset/HDR", "converted_exr")