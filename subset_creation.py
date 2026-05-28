import argparse
import json
import os
import shutil
import sys
from glob import glob
from typing import Iterable, Union
from tqdm import tqdm


def create_subset(src_root: str, dst_root: str, subset_ids: Iterable[Union[int, str]]) -> None:
    """Create a dataset subset by copying selected files and filtering metadata."""
    subset_ids = [str(ID) for ID in subset_ids]

    folders = [
        "Bbox",
        "Depth",
        "HDR_EXR",
        "InstanceSegmentationMasks",
        "NormalMap",
        "PanopticSegmentationMasks",
        "PNG",
        "SemanticSegmentationMasks",
    ]

    # Create folder structure
    for folder in folders:
        os.makedirs(os.path.join(dst_root, folder), exist_ok=True)

    # Copy files for each requested ID
    for ID in tqdm(subset_ids):

        mapping = [
            ("Bbox", f"bbox_{ID}.png"),
            ("Depth", f"d_{ID}.exr"),
            ("NormalMap", f"n_{ID}.exr"),
            ("PanopticSegmentationMasks", f"mSemPanoptic_{ID}_coarse.png"),
            ("PanopticSegmentationMasks", f"mSemPanoptic_{ID}_fine.png"),
            ("SemanticSegmentationMasks", f"mSemSegmentation_{ID}_coarse.png"),
            ("SemanticSegmentationMasks", f"mSemSegmentation_{ID}_fine.png"),
        ]

        for folder, filename in mapping:
            src = os.path.join(src_root, folder, filename)
            dst = os.path.join(dst_root, folder, filename)

            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                print(f"Missing: {src}")

        for pattern in [
            os.path.join(src_root, "HDR_EXR", f"h_{ID}_*.exr"),
            os.path.join(src_root, "PNG", f"o_{ID}_*.png"),
        ]:
            for src in glob(pattern):
                folder = os.path.basename(os.path.dirname(src))
                filename = os.path.basename(src)
                dst = os.path.join(dst_root, folder, filename)
                shutil.copy2(src, dst)

        pattern = os.path.join(
            src_root,
            "InstanceSegmentationMasks",
            f"mSemInstance_{ID}_instance_*.png",
        )

        for src in glob(pattern):
            filename = os.path.basename(src)
            dst = os.path.join(dst_root, "InstanceSegmentationMasks", filename)
            shutil.copy2(src, dst)

    # Filter JSON metadata
    json_path = os.path.join(src_root, "all_captures.json")
    with open(json_path, "r") as f:
        data = json.load(f)

    subset_data = {ID: data[ID] for ID in subset_ids if ID in data}
    with open(os.path.join(dst_root, "all_captures.json"), "w") as f:
        json.dump(subset_data, f, indent=4)

    # Copy static files
    static_files = [
        "gt_Coarse_color_mapping.json",
        "gt_Coarse_labelIds_mapping.json",
        "gt_Fine_color_mapping.json",
        "gt_Fine_labelIds_mapping.json",
        "rdr2_lookup_table.json",
    ]

    for filename in static_files:
        shutil.copy2(
            os.path.join(src_root, filename),
            os.path.join(dst_root, filename),
        )

    print("Subset creation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an RDR2 dataset subset.")
    parser.add_argument("src_root", help="Source dataset root")
    parser.add_argument("dst_root", help="Destination root for subset")
    parser.add_argument(
        "subset_ids",
        nargs="+",
        type=int,
        help="Subset capture IDs to include",
    )
    args = parser.parse_args()
    create_subset(args.src_root, args.dst_root, args.subset_ids)
