import json
import optuna
from dataset_handler.frag import init_data_module
from models_handler.transformer.kl_vit import KlVIT
from training.optuna_hyper import the_chosen
from utility.utility import BackboneType

if __name__ == "__main__": 
    data_module = init_data_module(
        data_dir="dataset",
        batch_size=35, 
        num_workers=12,
        sampler=True, 
        use_test=False, 
        use_masked_vit=False 
    )

    for p_plus in [False, True]:
        study = optuna.create_study(direction="minimize")  
        study.optimize(
            func=the_chosen(
                model_type=BackboneType.VIT_16, 
                datamodule=data_module, 
                num_epoch=25, 
                backbone_class=KlVIT,
                out_dir="FINAL_CLS_LOSS_KL_LOSS",
                device="cpu",
                p_plus=p_plus,
                model_params={
                    "backbone_type": "VIT_16",
                    "contrastive_loss": False,
                    "head_type": "CLS_SINGLE",
                    "k_classes": 11,
                    "lr": 0.0001584460127273449,
                    "min_epochs_head": 1,
                    "use_weighted_loss": False,
                    "weight_decay": 4.037216149183175e-05,
                    "double_head": False
                }
            ), 
            n_trials=5, 
            n_jobs=2
        ) 

        out_path = f"FINAL_CLS_LOSS_KL_LOSS\\best_hype_temp_CLS_lr_wd_p_plus:{str(p_plus)}.json"  # better to include .pkl extension
        
        with open(out_path, "w") as f:
            json.dump(
                obj={
                    "data": study.best_params
                }, 
                fp=f, 
                indent=4
            )