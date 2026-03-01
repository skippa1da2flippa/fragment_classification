import os
import torchvision.io as io
from dataset_handler.frag import denormalize
import torch

def denormalization(in_dir: str, out_dir: str) -> None:
    # Loop over top-level directories (like "train", "val", "test")
    for split_name in os.listdir(in_dir):
        split_path = os.path.join(in_dir, split_name)
        if not os.path.isdir(split_path):
            continue  # skip if not a folder

        # Loop over style subdirectories (like "Greek", "Byzantine", etc.)
        for style_name in os.listdir(split_path):
            style_path = os.path.join(split_path, style_name)
            if not os.path.isdir(style_path):
                continue  # skip if not a folder

            # Loop over image files inside each style folder
            for image_name in os.listdir(style_path):
                image_path = os.path.join(style_path, image_name)
                if not os.path.isfile(image_path):
                    continue  # skip if not a file

                # ✅ Load image as a tensor [C, H, W]
                image_tensor = io.decode_image(image_path).float()

                # ✅ Apply denormalization (your custom function)
                denorm_image = denormalize(image_tensor)

                # ✅ Create output directory (mirroring input structure)
                out_style_dir = os.path.join(out_dir, split_name, style_name)
                os.makedirs(out_style_dir, exist_ok=True)

                # ✅ Save denormalized image
                out_image_path = os.path.join(out_style_dir, image_name)
                io.write_png(denorm_image.to(torch.uint8), out_image_path)

                print(f"✅ Saved: {out_image_path}")
