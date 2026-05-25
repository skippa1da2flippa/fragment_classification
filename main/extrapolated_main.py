import json
import os
import pickle
import optuna
import pytorch_lightning as pl
from dataset_handler.frag import init_data_module
from training.optuna_hyper import just_a_wrapper
from utility.utility import BackboneType, HeadType
from models_handler.transformer.vit import VitClassifier
from pytorch_lightning.loggers import CSVLogger


if __name__ == "__main__":
    data_module = init_data_module(
        data_dir="datasets\\Square_12_lama",
        batch_size=256, 
        num_workers=8,
        sampler=False, 
        use_test=False, 
        use_masked_vit=False, 
        use_contourn=False
    )

    # for headtype in [HeadType.CLS_SINGLE, HeadType.SEQ_ENSEMBLE]:

    #     study = optuna.create_study(direction="maximize")  
    #     study.optimize(
    #         func=just_a_wrapper(
    #             model_type=BackboneType.VIT_16, 
    #             datamodule=data_module, 
    #             num_epoch=25, 
    #             contrastive_loss=False, 
    #             head_type=headtype, 
    #             masked_attention=False, 
    #             out_dir="VALID_POMPAF_extr_llama", 
    #             optimization_mode="max"
    #         ), 
    #         n_trials=9, 
    #         n_jobs=1
    #     ) 

    #     # Print the best set of hyperparameters
    #     print(f"Best hyperparameters for CLS ce loss with head_type: {headtype.name} --->", study.best_params)
    #     out_path = f"VALID_POMPAF_extr_llama\\best_hype_{headtype.name}_max_acc.json"  # better to include .pkl extension
        
    #     with open(out_path, "w") as f:
    #         json.dump(
    #             obj={
    #                 "data": study.best_params,
    #                 "metric": study.best_value,
    #                 "trial": study.best_trial.number
    #             }, 
    #             fp=f, 
    #             indent=4
    #         )

    model = VitClassifier(
        backbone_type="VIT_16",
        lr=0.0007496232566924756,
        weight_decay=1.2936387874776099e-06,
        min_epochs_head=6,
        head_type="SEQ_ENSEMBLE",
        k_classes=4,  # Changed from 4 to 11 (number of art styles in dataset)
        use_weighted_loss=False, 
        contrastive_loss=False, 
        full_dataset=False,
        db_path="datasets\\Square_12_lama"
    )

    base = "final_VIT_POMPAFF_ALL"
    # CSV logger
    logger_csv = CSVLogger(
        save_dir=f"final_VIT_POMPAFF_ALL\\FULL_VIT_SQUARE_12_EXTR_logs",
        name=f"FINAL_VIT_csv_extrapolated_max_accuracy",
    )

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(base, f"FINAL_VIT_CHKT_VIT_SQUARE_12_EXTR"),
        filename=f"weights_extrapolated_SQUARE_12_max_accuracy",
        monitor="val_accuracy",
        mode="max",
        save_top_k=1
    )
    # early_stopping_cb = pl.callbacks.EarlyStopping(
    #     monitor="val_loss",     # Metric to monitor
    #     mode="min",             # "min" for loss, "max" for accuracy/F1
    #     patience=5,             # Number of epochs with no improvement
    #     min_delta=1e-4,         # Required improvement threshold
    #     verbose=False
    # )

    # -----------------------------
    # 🚀 Train
    # -----------------------------
    trainer = pl.Trainer(
        max_epochs=50,
        logger=logger_csv,
        callbacks=[checkpoint_cb], #, early_stopping_cb],
        enable_progress_bar=True,
        accelerator="auto",
        devices=1
    )

    trainer.fit(
        model=model, 
        datamodule=data_module
    )