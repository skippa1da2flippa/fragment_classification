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
    #     if headtype in [HeadType.NONE]:
    #         continue
       
    #     data_module = init_data_module(
    #         data_dir="Original_artworks",
    #         batch_size=256, 
    #         num_workers=12,
    #         sampler=False, 
    #         use_test=False, 
    #         use_masked_vit=False
    #     )
    #     study = optuna.create_study(direction="minimize")  
    #     study.optimize(
    #         func=just_a_wrapper(
    #             model_type=BackboneType.VIT_16, 
    #             datamodule=data_module, 
    #             num_epoch=25, 
    #             contrastive_loss=False, 
    #             head_type=headtype,
    #             masked_attention=False, 
    #             backbone_class=VitClassifier, 
    #             out_dir="original_artwork_experiment"
    #         ), 
    #         n_trials=5 , 
    #         n_jobs=1
    #     ) 
    #     print(f"Best hyperparameters for CLS ce loss with head_type: {headtype.name} --->", study.best_params)
    #     out_path = os.path.join(
    #         "original_artwork_experiment", 
    #         f"best_hype_{headtype.name}_head_upd_wo_CLS.json"
    #     )
        
    #     with open(out_path, "w") as f:
    #         json.dump(
    #             obj={
    #                 "data": study.best_params, 
    #                 "validation_loss": study.best_value
    #             }, 
    #             fp=f, 
    #             indent=4
    #         )

    backbone_type = "VIT_16"
    contrastive_loss = False
    full_dataset = True
    head_type = "SEQ_ENSEMBLE"
    k_classes = 11
    lr = 0.0011761586991960145
    masked_attention = False
    min_epochs_head = 10
    num_head_mha = 12
    use_weighted_loss = False
    weight_decay = 1.2349311158078914e-05

    data_module = init_data_module(
        data_dir="Original_artworks",
        batch_size=256, 
        num_workers=12,
        sampler=False, 
        use_test=full_dataset, 
        use_masked_vit=False
    )

    model = VitClassifier(
        backbone_type=backbone_type,
        lr=lr,
        weight_decay=weight_decay,
        min_epochs_head=min_epochs_head,
        head_type=head_type,
        k_classes=k_classes, 
        use_weighted_loss=use_weighted_loss, 
        contrastive_loss=contrastive_loss, 
        masked_attention=masked_attention
    )

    base = "final_VIT"
    # CSV logger
    logger_csv = CSVLogger(
        save_dir=f"final_VIT\\FULL_VIT_TEST_logs",
        name=f"FINAL_VIT_csv_original_artworks",
    )

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(base, f"FINAL_VIT_CHKT"),
        filename=f"weights_original_artworks",
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
        max_epochs=45,
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