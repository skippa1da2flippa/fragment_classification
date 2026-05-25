import os
import shutil
import pandas as pd


csv_path = "80_20_split.csv"
source_dir = "C:/Users/skippa/Documents/projects/fragment_classification/datasets/Square_160_lama_all"
output_dir = "C:/Users/skippa/Documents/projects/fragment_classification/datasets/Square_160_lama"

df = pd.read_csv(csv_path)

for split in ["train", "test"]:
    split_path = os.path.join(output_dir, split)
    os.makedirs(split_path)

for x, row in df.iterrows():
    style = row["style"]
    image_name = row["image"]
    split = row["split"]

    # source folder za dati stil
    style_src_dir = os.path.join(source_dir, style)

    if not os.path.exists(style_src_dir):
        continue

    # target folder
    target_dir = os.path.join(output_dir, split, style)
    os.makedirs(target_dir, exist_ok=True)

    # nadji sve fragmente koji pripadaju toj slici
    for file in os.listdir(style_src_dir):
        if file.startswith(image_name):
            src_file = os.path.join(style_src_dir, file)
            dst_file = os.path.join(target_dir, file)

            shutil.copy2(src_file, dst_file)

print("Gotovo!")