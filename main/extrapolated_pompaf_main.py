from dataset_handler.frag import init_data_module
from pytorch_lightning.loggers import CSVLogger
from models_handler.frenziness.gnn import UltimateGraphApproach
from models_handler.transformer.vit import VitClassifier, transfer_learning_load
import pytorch_lightning as pl
import os 

if __name__ == '__main__':
    data_module = init_data_module(
        data_dir="datasets\\CrossingCutsSplit_lama",
        batch_size=200, 
        num_workers=12,
        sampler=False, 
        use_test=False, 
        use_masked_vit=False, 
        use_contourn=True
    )
    
    # Create the model with the current set of hyperparameters
    model = VitClassifier(
        backbone_type="VIT_16",
        lr=0.0007496232566924756,
        weight_decay=1.2936387874776099e-06,
        min_epochs_head=6,
        head_type="SEQ_ENSEMBLE",
        k_classes=4, 
        use_weighted_loss=False, 
        contrastive_loss=False, 
        masked_attention=False, 
        full_dataset=False, 
        db_path="datasets\\CrossingCutsSplit_lama"
    )

    base = "POMPAF_final_VIT"
    # CSV logger
    logger_csv = CSVLogger(
        save_dir=f"{base}\\FULL_VIT_TEST_EXTR_logs",
        name=f"FINAL_VIT_csv_max_acc",
    )

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(base, f"FINAL_VIT_EXTR_CHKT"),
        filename=f"weights_max_acc",
        monitor="val_accuracy",
        mode="max",
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
        max_epochs=50,
        logger=logger_csv,
        callbacks=[checkpoint_cb],# early_stopping_cb],
        enable_progress_bar=True,
        accelerator="auto",
        devices=1
    )

    trainer.fit(
        model=model, 
        datamodule=data_module
    )

    model = transfer_learning_load(
        k_classes=4,
        weights_pth="final_VIT\\FINAL_VIT_CHKT_VIT_CLEOPATRA_EXTR_LAMA\\weights_extrapolated_max_accuracy-v2.ckpt",
        hparams_pth="final_VIT\\FULL_VIT_CLEOPATRA_EXTR_LAMA_logs\\FINAL_VIT_csv_extrapolated_max_accuracy\\version_5\\hparams.yaml"
    )

    base = "POMPAF_final_VIT"
    # CSV logger
    logger_csv = CSVLogger(
        save_dir=f"{base}\\FULL_VIT_TEST_EXTR_TF_logs",
        name=f"FINAL_VIT_csv_max_acc",
    )

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(base, f"FINAL_VIT_EXTR_TF_CHKT"),
        filename=f"weights_max_acc",
        monitor="val_accuracy",
        mode="max",
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
        max_epochs=50,
        logger=logger_csv,
        callbacks=[checkpoint_cb],# early_stopping_cb],
        enable_progress_bar=True,
        accelerator="auto",
        devices=1
    )

    trainer.fit(
        model=model, 
        datamodule=data_module
    )