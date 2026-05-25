import json
import os
import optuna
from pytorch_lightning.loggers import CSVLogger
from dataset_handler.frag import init_data_module
from models_handler.transformer.kl_vit import KlVIT
from training.optuna_hyper import the_chosen
from utility.utility import BackboneType
import pytorch_lightning as pl

# BEST PATHS:
# NORMAL: FINAL_MODELS\final_VIT_max_accuracy\FULL_VIT_TEST_logs\FINAL_VIT_KL_csv_max_acc\version_5\hparams.yaml
# MSKD: FINAL_MODELS\final_VIT_max_accuracy\FULL_VIT_TEST_logs\FINAL_VIT_KL_mskd_csv_max_acc\version_4\hparams.yaml
# EXTR: FINAL_MODELS\final_VIT_max_accuracy\FULL_VIT_TEST_logs\FINAL_VIT_KL_extr_csv_max_acc\version_0\hparams.yaml

if __name__ == "__main__": 
    data_module = init_data_module(
        data_dir="datasets\\Square_12_lama",
        batch_size=256, 
        num_workers=12,
        sampler=False, 
        use_test=False, 
        use_masked_vit=False, 
        use_contourn=False
    )

    model = KlVIT(
    #     **{
    #         "lr": 0.0006375877711597772,
    #         "weight_decay": 0.001352259206961784,
    #         "use_weighted_loss": True,
    #         "alpha": 0.4, # TODO rimetti 3
    #         "beta": 0.6,  # TODO rimetti 1.5
    #         "kl_symmetric": False, # TODO rimetti true
    #         "kl_reduction": "sum",
    #         "min_epochs_head": 2,
    #         "backbone_type": "VIT_16",
    #         "ce_minimum_epoch": 2,
    #         "temperature": 6.0,
    #         "head_type": "SEQ_ENSEMBLE",
    #         "double_head": False,
    #         "k_classes": 4,
    #         "full_dataset":  False, 
    #         "db_path":  "datasets\\CrossingCutsSplit"
    #     }

        # **{
        #     "lr": 0.00025216319989695046,
        #     "weight_decay": 6.129110631586137e-05,
        #     "use_weighted_loss": False,
        #     "alpha": 0.4, # TODO rimetti 3
        #     "beta": 0.6, # TODO rimetti 2
        #     "kl_symmetric": False, # TODO rimetti true
        #     "kl_reduction": "sum",
        #     "min_epochs_head": 2,
        #     "ce_minimum_epoch": 3,
        #     "temperature": 6.0,
        #     "head_type": "SEQ_ENSEMBLE",
        #     "double_head": False,
        #     "backbone_type": "VIT_16", 
        #     "masked_attention": True,
        #     "k_classes": 4,
        #     "full_dataset":  False, 
        #     "db_path":  "datasets\\CrossingCutsSplit"
        # },

        **{
            "lr": 0.00024583095587727533,
            "weight_decay": 0.002044727526480959,
            "use_weighted_loss": False,
            "alpha": 1.4,
            "beta": 3.6,
            "kl_symmetric": False,
            "kl_reduction": "sum",
            "min_epochs_head": 2,
            "ce_minimum_epoch": 2,
            "temperature": 6.0,
            "head_type": "SEQ_ENSEMBLE",
            "backbone_type": "VIT_16",
            "k_classes": 4,
            "full_dataset": True, 
            "db_path":  "datasets\\Square_12_lama"
        }
    )

    base = "final_VIT"
    # CSV logger
    logger_csv = CSVLogger(
        save_dir=f"final_VIT\\FULL_VIT_TEST_POMPAF_SQUARE12_EXT_logs",
        name=f"FINAL_VIT_KL_POMPAF_SQUARE12_EXT_csv_max_acc",
    )

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(base, f"FINAL_VIT_POMPAF_SQUARE12_EXT_CHKT"),
        filename=f"weights_KL_POMPAF_SQUARE12_EXT_max_acc",
        monitor="val_accuracy",
        mode="max",
        save_top_k=1
    )
    early_stopping_cb = pl.callbacks.EarlyStopping(
        monitor="val_accuracy",     # Metric to monitor
        mode="max",             # "min" for loss, "max" for accuracy/F1
        patience=10,             # Number of epochs with no improvement
        min_delta=1e-2,         # Required improvement threshold
        verbose=False
    )

    # -----------------------------
    # 🚀 Train
    # -----------------------------
    trainer = pl.Trainer(
        max_epochs=50,
        logger=logger_csv,
        callbacks=[checkpoint_cb], # early_stopping_cb],
        enable_progress_bar=True,
        accelerator="auto",
        devices=1
    )

    trainer.fit(
        model=model, 
        datamodule=data_module
    )

    # for p_plus in [False]:
    #     study = optuna.create_study(direction="maximize")  
    #     study.optimize(
    #         func=the_chosen(
    #             model_type=BackboneType.VIT_16, 
    #             datamodule=data_module, 
    #             num_epoch=25, 
    #             backbone_class=KlVIT,
    #             out_dir="VALID_CLS_LOSS_KL_LOSS_FRAG",
    #             device="cuda",
    #             p_plus=p_plus,
    #             model_params={
    #                 "backbone_type": "VIT_16",
    #                 "contrastive_loss": False,
    #                 "k_classes": 11,
    #                 #"lr": 0.0001584460127273449,
    #                 #"min_epochs_head": 1,
    #                 #"use_weighted_loss": False,
    #                 #"weight_decay": 4.037216149183175e-05,
    #                 "double_head": False, 
    #                 "masked_attention": False
    #             }
    #         ), 
    #         n_trials=10 if not p_plus else 4, 
    #         n_jobs=1
    #     ) 

    #     out_path = f"VALID_CLS_LOSS_KL_LOSS_FRAG\\best_extr_{str(p_plus)}.json"  # better to include .pkl extension
        
    #     os.makedirs("VALID_CLS_LOSS_KL_LOSS_FRAG", exist_ok=True)
    #     with open(out_path, "w") as f:
    #         json.dump(
    #             obj={
    #                 "data": study.best_params,
    #                 "score": study.best_value
    #             }, 
    #             fp=f, 
    #             indent=4
    #         )