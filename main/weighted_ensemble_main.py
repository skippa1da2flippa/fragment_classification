import json

import optuna
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
from dataset_handler.frag import init_data_module_ensemble
from models_handler.ensemble.weighted_avg_ensemble import WeightedAverageEnsemble
from models_handler.transformer.vit import VitClassifier
import os

from training.optuna_hyper import ensemble_weighted_wrapper 


if __name__ == "__main__":

    data_module = init_data_module_ensemble(
        data_dirs=[
            "dataset", 
            "extrapolated_dataset"
        ], 
        num_workers=5, 
        batch_size=100, 
        use_test=True
    )

    model_paths = [
        (
            "final_VIT\\FINAL_VIT_CHKT\\weights.ckpt", 
            "final_VIT\\FULL_VIT_TEST_logs\\FINAL_VIT_csv\\version_0\\hparams.yaml"
        ), 
        (
            "final_VIT\\FINAL_VIT_CHKT\\weights_extrapolated.ckpt",
            "final_VIT\\FULL_VIT_TEST_logs\\FINAL_VIT_csv_extrapolated\\version_0\\hparams_extrapolated.yaml"
        ),
        (
            "final_VIT\\FINAL_VIT_CHKT\\weights_masked_head_upd.ckpt",
            "final_VIT\\FULL_VIT_TEST_logs\\FINAL_VIT_csv_masked_head_upd\\version_0\\hparams.yaml"
        )
    ]
    base_path = "Weighted_MLP_ENSEMBLE_NEW"
    study = optuna.create_study(direction="minimize")  
    study.optimize(
        func=ensemble_weighted_wrapper(
            datamodule=data_module, 
            model_paths=model_paths,
            model_types=VitClassifier,
            num_epoch=20,
            bs_path=base_path
        ), 
        n_trials=15, 
        n_jobs=1
    ) 

    print(f"Best hyperparameters for MLP", study.best_params)
    out_path = os.path.join(
        base_path, 
        f"best.json"
    )
    with open(out_path, "w") as f:
        json.dump(
            obj={
                "data": study.best_params
            }, 
            fp=f, 
            indent=4
        )

    # model = WeightedAverageEnsemble(
    #     model_paths=model_paths,
    #     model_types=VitClassifier,
    #     learners_name=[
    #         "base_vit", 
    #         "extr_vit",
    #         "mskd_vit"
    #     ], 
    #     mlp_num_layer=3, 
    #     model_dataset_info=[0, 1, 0]
    # )

    # base = "Weighted_ENSEMBLE"
    # # CSV logger
    # logger_csv = CSVLogger(
    #     save_dir=os.path.join(base, "Weighted_ENSEMBLE_logs"),
    #     name=f"Weighted_ENSEMBLE",
    # )

    # checkpoint_cb = pl.callbacks.ModelCheckpoint(
    #     dirpath=os.path.join(base, "Weighted_ENSEMBLE_CHKT"),
    #     filename=f"Weighted_ENSEMBLE",
    #     monitor="mlp_val_loss",
    #     mode="min",
    #     save_top_k=1
    # )
    # early_stopping_cb = pl.callbacks.EarlyStopping(
    #     monitor="mlp_val_loss", # Metric to monitor
    #     mode="min",             # "min" for loss, "max" for accuracy/F1
    #     patience=5,             # Number of epochs with no improvement
    #     min_delta=1e-4,         # Required improvement threshold
    #     verbose=False
    # )

    # # -----------------------------
    # # 🚀 Train
    # # -----------------------------
    # trainer = pl.Trainer(
    #     max_epochs=30,
    #     logger=logger_csv,
    #     callbacks=[checkpoint_cb, early_stopping_cb],
    #     enable_progress_bar=True,
    #     accelerator="auto",
    #     devices=1
    # )

    # trainer.fit(
    #     model=model, 
    #     datamodule=data_module
    # )


