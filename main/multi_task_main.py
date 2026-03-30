

import torch

from models_handler.transformer.multi_task_vit import MultiTaskVit
from training.full_pipeline import find_vit_hyper_and_train_model
from training.optuna_hyper import just_a_wrapper_multi_task
from utility.utility import BackboneType


if __name__ == "__main__":
    _ = find_vit_hyper_and_train_model(
        dataset_path="dataset",
        experiment_out_path="experiments_multi_task_vit",
        model_type=MultiTaskVit,
        optuna_function_wrapper=just_a_wrapper_multi_task,
        optuna_wrapper_kwargs={
            "backbone_class": MultiTaskVit,
            "masked_attention": False,
            "out_dir": "experiments_multi_task_vit",
            "model_type": BackboneType.VIT_16,
        },
        additional_model_params_f={
            "losses_weight": lambda x: torch.tensor([x, 1 - x]),
            "fusion_params": lambda trial: torch.tensor([
                trial["fusion_w_style"], 
                trial["fusion_w_epoch"], 
                trial["fusion_divisor"]
            ])
        },
        additional_param_key=["losses_weight", "fusion_params"],
        use_masked_vit=False, 
        use_countour=False
    )