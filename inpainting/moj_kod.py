import os
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import main

input_root = "C:/Users/skippa/Documents/projects/fragment_classification/datasets/fragment_dataset"
output_root = "C:/Users/skippa/Documents/projects/fragment_classification/datasets/fragment_dataset_ext_new"

valid_ext = (".jpg", ".png", ".jpeg", ".bmp")
splits = ["train"]


def collect_tasks():
    tasks = []

    for split in splits:
        split_path = os.path.join(input_root, split)

        if not os.path.exists(split_path):
            continue

        for class_name in os.listdir(split_path):
            if class_name not in ["Renaissance", "Roman", "Surrealism"]:
                continue

            class_path = os.path.join(split_path, class_name)

            if not os.path.isdir(class_path):
                continue

            out_class_path = os.path.join(output_root, split, class_name)
            os.makedirs(out_class_path, exist_ok=True)

            for file_name in os.listdir(class_path):
                if not file_name.lower().endswith(valid_ext):
                    continue

                img_path = os.path.join(class_path, file_name)
                out_path = os.path.join(out_class_path, file_name)

                tasks.append((img_path, out_path))

    return tasks


def process_one(task):
    img_path, out_path = task

    try:
        processed = main.impaint(img_path)
        processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)

        ok = cv2.imwrite(out_path, processed_bgr)

        if not ok:
            return img_path, False, "cv2.imwrite failed"

        return img_path, True, None

    except Exception as e:
        return img_path, False, str(e)


if __name__ == "__main__":
    tasks = collect_tasks()

    print(f"Pronađeno slika: {len(tasks)}")

    max_workers = min(8, os.cpu_count() or 1)
    errors = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures)):
            img_path, ok, error = future.result()

            if not ok:
                errors.append((img_path, error))

    print("✅ Gotovo! Obradjene slike su u:", output_root)

    if errors:
        print(f"⚠️ Broj grešaka: {len(errors)}")
        for img_path, error in errors[:20]:
            print(img_path, "->", error)