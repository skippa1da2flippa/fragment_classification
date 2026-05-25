from dataset_handler.frag import init_data_module
from pytorch_lightning.loggers import CSVLogger
from models_handler.frenziness.gnn import UltimateGraphApproach
from models_handler.transformer.vit import VitClassifier, transfer_learning_load
import pytorch_lightning as pl
import os 

if __name__ == '__main__':
    data_module = init_data_module(
        data_dir="datasets\\CrossingCutsSplit",
        batch_size=200, 
        num_workers=12,
        sampler=False, 
        use_test=False
    )
    
    # Create the model with the current set of hyperparameters
    model = VitClassifier(
        backbone_type="VIT_16",
        lr=0.0001584460127273449,
        weight_decay=4.037216149183175e-05,
        min_epochs_head=3,
        head_type="CLS_SINGLE",
        k_classes=4, 
        use_weighted_loss=False, 
        contrastive_loss=False
    )

    base = "POMPAF_final_VIT"
    # CSV logger
    logger_csv = CSVLogger(
        save_dir=f"{base}\\FULL_VIT_TEST_logs",
        name=f"FINAL_VIT_csv_max_acc",
    )

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(base, f"FINAL_VIT_CHKT"),
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
        weights_pth="FINAL_MODELS\\final_VIT_max_accuracy\\FINAL_VIT_CHKT\\weights_max_acc-v1.ckpt",
        hparams_pth="FINAL_MODELS\\final_VIT_max_accuracy\\FULL_VIT_TEST_logs\\FINAL_VIT_csv_max_acc\\version_1\\hparams.yaml"
    )

    base = "POMPAF_final_VIT"
    # CSV logger
    logger_csv = CSVLogger(
        save_dir=f"{base}\\FULL_VIT_TEST_TF_logs",
        name=f"FINAL_VIT_csv_max_acc",
    )

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(base, f"FINAL_VIT_TF_CHKT"),
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