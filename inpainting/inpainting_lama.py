import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from simple_lama_inpainting import SimpleLama


# =========================================================datasets\CrossingCutsSplit
# CONFIG
# =========================================================

INPUT_ROOT = Path(
    "C:/Users/skippa/Documents/projects/fragment_classification/datasets/Square_pieces/160_pieces"
)

OUTPUT_ROOT = Path(
    "C:/Users/skippa/Documents/projects/fragment_classification/datasets/Square_160_lama"
)

ALPHA_THRESHOLD = 128


# =========================================================
# CHECK CUDA
# =========================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available!")

print("CUDA DEVICE:", torch.cuda.get_device_name(0))


# =========================================================
# LOAD MODEL
# =========================================================

print("Loading LaMa model on GPU...")

lama = SimpleLama(device="cuda")

print("Model loaded.")


# =========================================================
# HELPERS
# =========================================================

def load_image_rgba_with_alpha_mask(path, alpha_threshold=128):
    img = Image.open(path).convert("RGBA")

    arr = np.array(img)

    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    # foreground
    fg_mask = (alpha >= alpha_threshold).astype(np.uint8)

    # remove hidden RGB under transparency
    rgb_clean = rgb * fg_mask[:, :, None]

    # mask for inpainting
    lama_mask = (alpha < alpha_threshold).astype(np.uint8) * 255

    return rgb_clean, lama_mask


def process_image(input_path, output_path):

    try:
        img_rgb, mask = load_image_rgba_with_alpha_mask(
            input_path,
            alpha_threshold=ALPHA_THRESHOLD
        )

        result = lama(
            Image.fromarray(img_rgb),
            Image.fromarray(mask).convert("L")
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        result.save(output_path)

    except Exception as e:
        print(f"[ERROR] {input_path}")
        print(e)


# =========================================================
# COLLECT FILES
# =========================================================

jobs = []

for root, _, files in os.walk(INPUT_ROOT):

    for file in files:

        if file.lower().endswith(".png"):

            input_path = Path(root) / file

            rel_path = input_path.relative_to(INPUT_ROOT)

            output_path = OUTPUT_ROOT / rel_path

            jobs.append((input_path, output_path))


print(f"Found {len(jobs)} images")


# =========================================================
# PROCESS
# =========================================================

for input_path, output_path in tqdm(jobs):

    process_image(input_path, output_path)


print("DONE")