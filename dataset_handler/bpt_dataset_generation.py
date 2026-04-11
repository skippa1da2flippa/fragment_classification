import os
from concurrent.futures import ThreadPoolExecutor
from utility.patch_shap_bpt import single_pipeline_bpt

def process_one_directory(input_dir: str, output_dir: str) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    for file in os.listdir(input_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):  # assuming image extensions
            img_path = os.path.join(input_dir, file)
            json_file = os.path.splitext(file)[0] + '.json'
            json_path = os.path.join(output_dir, json_file)
            single_pipeline_bpt(img_path, json_path)


def get_bpt_dataset(dataset_path: str, output_path: str) -> None:   
    for split in os.listdir(dataset_path):
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            continue
        
        styles = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for style in styles:
                input_dir = os.path.join(split_path, style)
                output_dir = os.path.join(output_path, split, style)
                futures.append(executor.submit(process_one_directory, input_dir, output_dir))
            
            for future in futures:
                future.result()  # wait for all to complete