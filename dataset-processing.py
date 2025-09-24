import sys

import imagecodecs
import os
import pyexr
import json
import numpy as np
import imageio
from skimage.morphology import remove_small_objects
from collections import defaultdict

import numpy as np

def make_json_serializable(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(v) for v in obj)
    elif isinstance(obj, (np.integer, )):
        return int(obj)
    elif isinstance(obj, (np.floating, )):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def save_value_for_entity(save_path, entity_filename, counter, name, value):
    all_json_path = os.path.join(save_path, "all_captures.json")

    with open(all_json_path, "r", encoding="utf-8") as f:
        all_captures = json.load(f)

    counter_str = str(counter)
    if counter_str in all_captures:
        capture_data = all_captures[counter_str]

        if "Entities" in capture_data and isinstance(capture_data["Entities"], list):
            updated = False
            for ent in capture_data["Entities"]:
                if ent.get("Filename") == entity_filename:
                    # Convert value to JSON-safe type
                    ent[name] = make_json_serializable(value)
                    updated = True
                    break

            if not updated:
                raise ValueError(f"Entity '{entity_filename}' not found in capture {counter_str}")
        else:
            raise ValueError(f"No 'Entities' list found for capture {counter_str}")
    else:
        raise ValueError(f"No capture found for counter {counter_str}")

    # Save JSON back
    with open(all_json_path, "w", encoding="utf-8") as f:
        json.dump(all_captures, f, indent=4)

def remove_entity(save_path, counter, entity_filename):

    all_json_path = os.path.join(save_path, "all_captures.json")

    with open(all_json_path, "r", encoding="utf-8") as f:
        all_captures = json.load(f)

    counter_str = str(counter)
    if counter_str in all_captures:
        capture_data = all_captures[counter_str]

        if "Entities" in capture_data and isinstance(capture_data["Entities"], list):
            original_len = len(capture_data["Entities"])

            capture_data["Entities"] = [
                e for e in capture_data["Entities"] if e.get("Filename") != entity_filename
            ]

            new_len = len(capture_data["Entities"])
            removed = original_len - new_len

            if removed > 0 and "Capture" in capture_data and "EntityCount" in capture_data["Capture"]:
                capture_data["Capture"]["EntityCount"] = max(0, capture_data["Capture"]["EntityCount"] - removed)
                print("changed entity count in capture ", counter)

                if capture_data["Capture"]["EntityCount"] == 0 or not capture_data["Entities"]:
                    #del all_captures[counter_str]
                    print(f"Capture {counter_str} has no entities left.")

        else:
            raise ValueError(f"No 'Entities' list found for {counter_str}.")
    else:
        print("Capture was deleted already.")


    with open(all_json_path, "w", encoding="utf-8") as f:
        json.dump(all_captures, f, indent=4)

def get_array_from_path(path: str):
    if path.endswith('.png'):
        img = imagecodecs.imread(path)
        return np.array(img)
    elif path.endswith('.exr'):
        img = pyexr.open(path).get()
        return np.array(img, dtype=np.float32)
    elif path.endswith('.jxr'):
        with open(path, 'rb') as fh:
            jpegxr = fh.read()
        numpy_array = imagecodecs.jpegxr_decode(jpegxr)
        return np.array(numpy_array)
    else:
        return None

def get_bbox_from_mask(mask, image_shape, counter, entity, save_path, depth_path, threshold=0):
    bbox_list_calculated = []
    invisible = False

    if not np.any(mask):
        remove_entity(save_path, counter, entity)
        invisible=True

    else:
        ys, xs = np.where(mask)
        new_x1 = max(xs.min() - threshold, 0)
        new_y1 = max(ys.min() - threshold, 0)
        new_x2 = min(xs.max() + threshold, image_shape[1])
        new_y2 = min(ys.max() + threshold, image_shape[0])
        bbox_list_calculated=[new_x1, new_y1, new_x2, new_y2]

        mask_area = np.count_nonzero(mask)
        image_area = image_shape[0] * image_shape[1]
        percentage = (mask_area / image_area) * 100
        print("Percentage of mask in image: ", percentage)
        save_value_for_entity(save_path, entity, counter, "PercentageMaskImage", percentage)


        mask_crop = mask[new_y1:new_y2, new_x1:new_x2]
        mask_area = np.count_nonzero(mask_crop)
        bbox_area = (new_y2 - new_y1) * (new_x2 - new_x1)
        percentage_bbox = (mask_area / bbox_area) * 100
        print("Percentage in Bbox: ", percentage_bbox)
        save_value_for_entity(save_path, entity, counter, "PercentageMaskBbox", percentage_bbox)


        double_check_file = os.path.join(save_path, "double-check.txt")
        if percentage == percentage_bbox or percentage > 70:
            with open(double_check_file, "a", encoding="utf-8") as f:
                f.write(f"Capture {counter}: Error at capture {depth_path} for entity {entity}\n - - percentage: {percentage}")
        elif percentage_bbox < 5:
            with open(double_check_file, "a", encoding="utf-8") as f:
                f.write(f"Capture {counter}: Error at capture {depth_path} for entity {entity}\n - Animal percentage in Bbox too small - percentage: {percentage_bbox}\n")


    return bbox_list_calculated, invisible

def apply_bbox_mask(mask: np.ndarray, bbox: list) -> np.ndarray:
    """
    Apply a bounding box to a 2D mask (0/1), returns masked 2D mask.
    """
    masked_mask = np.zeros_like(mask, dtype=np.uint8)
    if len(bbox) > 0:
        x_min, y_min, x_max, y_max = bbox
        masked_mask[y_min:y_max, x_min:x_max] = mask[y_min:y_max, x_min:x_max]
    return masked_mask

def bbox_to_cocoBbox(bbox):
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    return [x_min, y_min, width, height]

def image_dif(image1: np.ndarray, image2: np.ndarray, counter: int, entity, save_path, depth_path,
              thresh=1, color=None) -> tuple[np.ndarray, tuple]:
    """
    Computes the difference between two images and returns a Panoptic-ID mask.
    """
    invisible = False

    if color is None:
        raise ValueError("Color (Panoptic-ID) is None.")


    if image1.shape != image2.shape:
        raise ValueError("Images must have the same shape.")

    if image1.shape[2] == 4:
        image1 = image1[:, :, :3]
        image2 = image2[:, :, :3]

    image_shape = image1.shape

    dif = np.abs(image1 - image2)
    mask = np.all(dif > [thresh, thresh, thresh], axis=-1)

    cleaned_mask = remove_small_objects(mask, min_size=6)

    bbox, invisible = get_bbox_from_mask(cleaned_mask, image_shape, counter, entity, save_path, depth_path, 3)
    final_mask = apply_bbox_mask(cleaned_mask.astype(np.uint8), bbox)
    final_mask_uint = final_mask.astype(np.uint32) * color

    final_bbox, invisible = get_bbox_from_mask(final_mask, image_shape, counter, entity, save_path, depth_path, 5)
    if not invisible:
        #final_bbox_coco = bbox_to_cocoBbox(final_bbox)
        #save_value_for_entity(save_path, entity, counter, "BoundingBoxCalculated", final_bbox_coco)
        save_value_for_entity(save_path, entity, counter, "BoundingBoxCalculated", final_bbox)

    return final_mask_uint, final_bbox, invisible

def select_pixels_by_color_sum(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Compare two HxWx3 images and select the pixel with the higher color sum at each location.

    Args:
        img1 (np.ndarray): First image array.
        img2 (np.ndarray): Second image array.

    Returns:
        np.ndarray: New image array with selected pixels.
    """
    # Compute the sum across the color channels for each pixel
    mask = img1 >= img2
    output = np.where(mask, img1, img2)
    return output

def colormask_to_trainid(mask_rgb: np.ndarray, color_mapping, id_mapping) -> np.ndarray:
    h, w, _ = mask_rgb.shape
    trainid_mask = np.zeros((h, w), dtype=np.uint8)

    for cls, color in color_mapping.items():
        color_255 = np.array(color, dtype=np.uint8)
        matches = np.all(mask_rgb == color_255, axis=-1)
        trainid_mask[matches] = id_mapping[cls]

    return trainid_mask


def map_class_name(class_name, segmentation_type):
    with open("rdr2_classes_processed.json", "r", encoding="utf-8") as f:
        class_hierarchy = json.load(f)
    for coarse_name, fine_dict in class_hierarchy.items():
        for fine_name, model_names in fine_dict.items():
            if class_name in model_names:
                if segmentation_type == "fine":
                    return fine_name
                elif segmentation_type == "coarse":
                    return coarse_name

    return class_name


def generate_screenshot_mask(
        base_path: str,
        save_path: str,
        depth_image_list,
        counter: int,
        depth_path,
        folder_path,
        segmentation_type,
        class_to_color=None,
        class_to_id=None,
        threshold=1,
        save_png=False) -> tuple:
    print("Counter: ", counter)
    depth_path = base_path + '/Depth'
    save_path_instance = save_path + "/InstanceSegmentationMasks"
    reference_image = get_array_from_path(depth_path + "/" + depth_image_list[0])

    masks = []
    bboxes = []

    result_mask = np.zeros(reference_image.shape[:2], dtype=np.uint32)
    class_instance_counter = defaultdict(int)
    invisible = False
    instance_counter = 0

    for i, image in enumerate(depth_image_list[1:]):

        image_array = get_array_from_path(depth_path + "/" + image)
        name_part = image.split("_", 1)[-1]
        entity = depth_image_list[i + 1].rsplit(".", 1)[0]
        class_name = name_part.rsplit(".", 1)[0]
        class_name_mapped = map_class_name(class_name, segmentation_type)
        print("********************************")
        print("entity: ", entity)
        print("********************************")

        class_id = class_to_id[class_name_mapped]

        # Start instance number at 1
        if class_instance_counter[class_name] == 0:
            class_instance_counter[class_name] += 1
        if class_instance_counter[class_name] > 99:
            raise ValueError(
                f"At most 99 instances per class in each image allowed! '{class_name}' has {class_instance_counter[class_name]} instances already."
            )

        instance_id_per_class = class_instance_counter[class_name]
        color = class_id * 100 + instance_id_per_class
        mask, bbox, invisible = image_dif(reference_image, image_array, counter, entity, save_path, depth_path, thresh=threshold, color=color)
        print("Invivible: ", invisible, "Segmentation type: ", segmentation_type)
        if not invisible and segmentation_type == "coarse":
            mask_instance = (mask == color).astype(np.uint8)
            print(f"saving {entity} as mSemInstance_{counter}_instance_{instance_counter}.png")
            imageio.imwrite(save_path_instance + f'/mSemInstance_{counter}_instance_{instance_counter}.png', mask_instance)
            instance_counter += 1

        masks.append(mask)
        bboxes.append(bbox)
        result_mask = select_pixels_by_color_sum(result_mask, mask)

        print("identity: ", class_name)
        print("invisible: ", invisible)
        print("instance count: ", instance_id_per_class)

        if not invisible:
            class_instance_counter[class_name] += 1


        print("Generated mask " + str(i + 1) + "/" + str(len(depth_image_list)-1))


    array_to_save = result_mask.copy()

    if not np.any(array_to_save):
        save_png = False
    if save_png:
        array_to_save[array_to_save == 0] = 25500
        print(np.unique(array_to_save))
        save_path_panoptic = save_path + "/PanopticSegmentationMasks"
        #panoptic_rgb = panoptic_id_to_rgb(array_to_save)
        imageio.imwrite(save_path_panoptic + f'/mSemPanoptic_{counter}_{segmentation_type}.png', array_to_save.astype(np.uint16))
        semantic_segmentation_array = array_to_save // 100
        print(np.unique(semantic_segmentation_array))
        save_path_segmentation = save_path + "/SemanticSegmentationMasks"
        imageio.imwrite(save_path_segmentation + f'/mSemSegmentation_{counter}_{segmentation_type}.png', semantic_segmentation_array.astype(np.uint8))
        #imageio.imwrite(save_path + f'/mSemPanoptic_{counter}.tiff', array_to_save.astype(np.uint32), format='TIFF')
    return array_to_save, masks, bboxes, save_png

"""
def panoptic_id_to_rgb(mask_uint32):
    R = (mask_uint32 >> 16).astype(np.uint8)
    G = ((mask_uint32 >> 8) & 255).astype(np.uint8)
    B = (mask_uint32 & 255).astype(np.uint8)
    rgb = np.stack([R, G, B], axis=-1)
    return rgb"""

"""
def rgb_to_panoptic_id(rgb_mask):
    return (rgb_mask[:,:,0].astype(np.uint32) << 16) + \
           (rgb_mask[:,:,1].astype(np.uint32) << 8) + \
           rgb_mask[:,:,2].astype(np.uint32)"""

