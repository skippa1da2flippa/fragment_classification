import json
import os
import optuna
from dataset_handler.frag import init_data_module
from training.optuna_hyper import just_a_wrapper
from utility.utility import BackboneType, HeadType
from models_handler.transformer.vit import VitClassifier
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl

if __name__ == "__main__":

    # for headtype in HeadType:
    #     if headtype in [HeadType.CLS_SINGLE, HeadType.NONE, HeadType.SEQ_ENSEMBLE_MAX, HeadType.SEQ_ENSEMBLE_CLS]:
    #         continue

    #     for val in [False, True]:
    #         data_module = init_data_module(
    #             data_dir="dataset",
    #             batch_size=256, 
    #             num_workers=12,
    #             sampler=False, 
    #             use_test=False, 
    #             use_masked_vit=True, 
    #             use_contourn=val
    #         )
    #         study = optuna.create_study(direction="minimize")  
    #         study.optimize(
    #             func=just_a_wrapper(
    #                 model_type=BackboneType.VIT_16, 
    #                 datamodule=data_module, 
    #                 num_epoch=25, 
    #                 contrastive_loss=False, 
    #                 head_type=headtype,
    #                 masked_attention=True, 
    #                 backbone_class=VitClassifier, 
    #                 out_dir="masked_experiment"
    #             ), 
    #             n_trials=5 , 
    #             n_jobs=1
    #         ) 
    #         print(f"Best hyperparameters for CLS ce loss with head_type: {headtype.name} --->", study.best_params)
    #         out_path = os.path.join(
    #             "masked_experiment", 
    #             "masked_best_params",
    #             f"best_hype_{headtype.name}_head_upd_wo_CLS.json"
    #         )
            
    #         with open(out_path, "w") as f:
    #             json.dump(
    #                 obj={
    #                     "data": study.best_params, 
    #                     "validation_loss": study.best_value
    #                 }, 
    #                 fp=f, 
    #                 indent=4
    #             )

    backbone_type = "VIT_16"
    contrastive_loss = False
    head_type = "SEQ_ENSEMBLE"
    k_classes = 11
    lr = 7.219872467151126e-05
    masked_attention = True
    min_epochs_head = 7
    num_head_mha = 12
    use_weighted_los = True
    weight_decay = 0.0004909363971039878

    data_module = init_data_module(
        data_dir="dataset",
        batch_size=256, 
        num_workers=12,
        sampler=False, 
        use_test=True, 
        use_masked_vit=True, 
        use_contourn=True
    )

    model = VitClassifier(
        backbone_type="VIT_16",
        lr=0.00030593439210768514,
        weight_decay=0.00021816569175271556,
        min_epochs_head=4,
        head_type="SEQ_ENSEMBLE",
        k_classes=11, 
        use_weighted_loss=True, 
        contrastive_loss=False, 
        masked_attention=masked_attention
    )

    base = "final_VIT"
    # CSV logger
    logger_csv = CSVLogger(
        save_dir=f"final_VIT\\FULL_VIT_TEST_logs",
        name=f"FINAL_VIT_csv_masked_head_upd_wo_CLS",
    )

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(base, f"FINAL_VIT_CHKT"),
        filename=f"weights_masked_head_upd_wo_CLS",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )
    early_stopping_cb = pl.callbacks.EarlyStopping(
        monitor="val_loss",     # Metric to monitor
        mode="min",             # "min" for loss, "max" for accuracy/F1
        patience=5,             # Number of epochs with no improvement
        min_delta=1e-4,         # Required improvement threshold
        verbose=False
    )

    # -----------------------------
    # 🚀 Train
    # -----------------------------
    trainer = pl.Trainer(
        max_epochs=30,
        logger=logger_csv,
        callbacks=[checkpoint_cb, early_stopping_cb],
        enable_progress_bar=True,
        accelerator="auto",
        devices=1
    )

    trainer.fit(
        model=model, 
        datamodule=data_module
    )