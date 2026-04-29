from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
from dataset_handler.frag import init_data_module_ensemble
from models_handler.ensemble.baseline_ensemble import BaselineEnsemble
from models_handler.transformer.vit import VitClassifier
import os


if __name__ == "__main__":

    base_dataset_path: str = "datasets"
    mask_on_learner: int = 2

    data_module = init_data_module_ensemble(
        data_dirs=[
            os.path.join(base_dataset_path, "fragment_dataset"),
            os.path.join(base_dataset_path, "extrapolated_dataset")
        ], 
        num_workers=10, 
        batch_size=50, 
        use_test=True
    )

    model_path = [
        (
            "final_VIT\\FINAL_VIT_CHKT\\weights_max_acc-v1.ckpt", 
            "final_VIT\\FULL_VIT_TEST_logs\\FINAL_VIT_csv_max_acc\\version_1\\hparams.yaml"
        ), 
        (
            "final_VIT\\FINAL_VIT_CHKT\\weights_extrapolated_max_accuracy-v1.ckpt",
            "final_VIT\\FULL_VIT_TEST_logs\\FINAL_VIT_csv_extrapolated_max_accuracy\\version_2\\hparams.yaml"
        ),
        (
            "final_VIT\\FINAL_VIT_CHKT\\weights_masked_head_upd_wo_CLS_max_acc.ckpt",
            "final_VIT\\FULL_VIT_TEST_logs\\FINAL_VIT_csv_masked_head_upd_wo_CLS_max_acc\\version_1\\hparams.yaml"
        )
    ]

    model = BaselineEnsemble(
        model_paths=model_path,
        model_types=VitClassifier,
        learners_name=[
            "base_vit", 
            "extr_vit",
            "mskd_vit"
        ],
        model_dataset_info=[0, 1, 0], 
        mask_on_learner=mask_on_learner
    )

    base = "Baseline_ENSEMBLE"
    # CSV logger
    logger_csv = CSVLogger(
        save_dir=os.path.join(base, "Baseline_ENSEMBLE_logs"),
        name=f"Baseline_ENSEMBLE",
    )

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(base, "Baseline_ENSEMBLE_CHKT"),
        filename=f"Baseline_ENSEMBLE",
        monitor="aggregator_val_loss",
        mode="min",
        save_top_k=1
    )
    early_stopping_cb = pl.callbacks.EarlyStopping(
        monitor="aggregator_val_loss", # Metric to monitor
        mode="min",             # "min" for loss, "max" for accuracy/F1
        patience=5,             # Number of epochs with no improvement
        min_delta=1e-4,         # Required improvement threshold
        verbose=False
    )

    # -----------------------------
    # 🚀 Train
    # -----------------------------
    trainer = pl.Trainer(
        max_epochs=10,
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