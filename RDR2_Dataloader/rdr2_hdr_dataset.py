import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import re
import random
import cv2

class HDRDataset(Dataset):
    def __init__(self, root_dir, json_path, split="train",
                 filter_fn=None, daytime_filter=None, transform=None):
        """
        root_dir: dataset root folder containing HDR/, PNG/, HDR_np/
        json_path: path to all_captures.json
        split: "train", "val", "test"
        filter_fn: callable(capture_meta) -> bool
        daytime_filter: list of allowed daytime indices (e.g., [0, 7, 12])
        transform: transform for LDR images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        with open(json_path, "r") as f:
            meta = json.load(f)

        # Regex to extract scene_id and daytime_index from filenames
        pattern = re.compile(r"[oh]_(\d+)_(\d+)\.")

        # Loop over PNG folder to discover samples
        png_dir = os.path.join(root_dir, "PNG")
        for fname in os.listdir(png_dir):
            if not fname.endswith(".png"):
                continue

            m = pattern.match(fname)
            if not m:
                continue
            scene_id, daytime_idx = m.groups()
            scene_id = str(scene_id)
            daytime_idx = int(daytime_idx)

            if scene_id not in meta:
                continue

            capture_meta = meta[scene_id]["Capture"]

            # Metadata filter
            if filter_fn and not filter_fn(capture_meta):
                continue

            # Daytime filter (by index number, not string category)
            if daytime_filter and daytime_idx not in daytime_filter:
                continue

            png_path = os.path.join(png_dir, fname)
            hdr_path = os.path.join(root_dir, "HDR_np", f"h_{scene_id}_{daytime_idx}.npy")

            if os.path.exists(png_path) and os.path.exists(hdr_path):
                self.data.append({
                    "scene_id": scene_id,
                    "daytime_idx": daytime_idx,
                    "png": png_path,
                    "hdr": hdr_path,
                    "meta": capture_meta
                })

        # Apply split
        self.data = self._split_data(self.data, split)

    def _split_data(self, data, split, seed=42):
        # Split by scene_id (so same scene doesn’t leak across sets)
        scene_ids = list(set(d["scene_id"] for d in data))

        # Shuffle deterministically for reproducibility
        rng = random.Random(seed)
        rng.shuffle(scene_ids)

        n = len(scene_ids)
        if split == "train":
            selected = set(scene_ids[: int(0.8 * n)])
        elif split == "val":
            selected = set(scene_ids[int(0.8 * n): int(0.95 * n)])
        else:  # test
            selected = set(scene_ids[int(0.95 * n):])

        return [d for d in data if d["scene_id"] in selected]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        # Load LDR
        ldr = Image.open(entry["png"]).convert("RGB")
        if self.transform:
            ldr = self.transform(ldr)
        else:
            ldr = torch.from_numpy(np.array(ldr)).permute(2, 0, 1).float() / 255.0

        # Load HDR EXR
        hdr = cv2.imread(entry["hdr"], cv2.IMREAD_UNCHANGED)  # H x W x C, float32
        hdr = torch.from_numpy(hdr).permute(2, 0, 1).float()  # C x H x W
        
        return {
            "ldr": ldr,
            "hdr": hdr,
            "scene_id": entry["scene_id"],
            "daytime_idx": entry["daytime_idx"],
            "meta": entry["meta"]
        }
