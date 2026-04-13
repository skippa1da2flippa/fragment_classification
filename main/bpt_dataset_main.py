import os
from dataset_handler.bpt_dataset_generation import get_bpt_dataset


if __name__ == "__main__":
    base_dataset: str = "datasets"
    for db_name in os.listdir(base_dataset):
        if db_name not in ["extrapolated_dataset", "fragment_dataset"]:
            continue
        db_path: str = os.path.join(base_dataset, db_name)
        out_db_path: str = os.path.join(
            base_dataset, 
            "BPT_" + db_name
        )
        get_bpt_dataset(
            dataset_path=db_path, 
            output_path=out_db_path
        )

        print(f"\n\n !!!! Dataset {db_name} completed !!!!! \n\n")