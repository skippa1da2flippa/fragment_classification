import json

import optuna

from dataset_handler.frag import init_data_module
from training.optuna_hyper import graph_attention_vit_wrapper
from utility.utility import BackboneType, HeadType


if __name__ == "__main__":

    data_module = init_data_module(
        data_dir="datasets_test\\cleopatra_mock_test",
        batch_size=20, 
        num_workers=8,
        sampler=False, 
        use_test=False, 
        use_masked_vit=False, 
        use_contourn=False, 
        bpt_paths="datasets_test\\db_bpt_normal_test"
    )

    for headtype in [HeadType.CLS_SINGLE, HeadType.SEQ_ENSEMBLE]:

            study = optuna.create_study(direction="maximize")  
            study.optimize(
                func=graph_attention_vit_wrapper(
                    model_type=BackboneType.VIT_16, 
                    datamodule=data_module, 
                    num_epoch=25, 
                    head_type=headtype, 
                    masked_attention=False, 
                    out_dir="VALID_CLEOPATRA", 
                    optimization_mode="max",
                    db_path="datasets_test\\cleopatra_mock_test"
                ), 
                n_trials=9, 
                n_jobs=1
            ) 

            # Print the best set of hyperparameters
            print(f"Best hyperparameters for CLS ce loss with head_type: {headtype.name} --->", study.best_params)
            out_path = f"VALID_CLEOPATRA\\best_hype_{headtype.name}_max_acc.json"  # better to include .pkl extension
            
            with open(out_path, "w") as f:
                json.dump(
                    obj={
                        "data": study.best_params,
                        "metric": study.best_value,
                        "trial": study.best_trial.number
                    }, 
                    fp=f, 
                    indent=4
                )



