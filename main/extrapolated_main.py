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
        data_dir="extrapolated_dataset",
        batch_size=256, 
        num_workers=6,
        sampler=False, 
        use_test=True, 
        use_masked_vit=False 
    )

    # for headtype in HeadType:
    #     study = optuna.create_study(direction="minimize")  
    #     study.optimize(
    #         func=just_a_wrapper(
    #             model_type=BackboneType.VIT_16, 
    #             datamodule=data_module, 
    #             num_epoch=25, 
    #             contrastive_loss=False, 
    #             head_type=headtype, 
    #             masked_attention=False
    #         ), 
    #         n_trials=5, 
    #         n_jobs=1
    #     ) 

    #     # Print the best set of hyperparameters
    #     print(f"Best hyperparameters for CLS ce loss with head_type: {headtype.name} --->", study.best_params)
    #     out_path = f"extrapolated_best_params\\best_hype_{headtype.name}.json"  # better to include .pkl extension
        
    #     with open(out_path, "w") as f:
    #         json.dump(
    #             obj={
    #                 "data": study.best_params
    #             }, 
    #             fp=f, 
    #             indent=4
    #         )

    model = VitClassifier(
        backbone_type="VIT_16",
        lr=0.00030593439210768514,
        weight_decay=0.00021816569175271556,
        min_epochs_head=4,
        head_type="SEQ_ENSEMBLE",
        k_classes=11, 
        use_weighted_loss=True, 
        contrastive_loss=False
    )

    base = "final_VIT"
    # CSV logger
    logger_csv = CSVLogger(
        save_dir=f"final_VIT\\FULL_VIT_TEST_logs",
        name=f"FINAL_VIT_csv_extrapolated",
    )

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(base, f"FINAL_VIT_CHKT"),
        filename=f"weights_extrapolated",
        monitor="val_loss",
        mode="min",
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
        max_epochs=30,
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