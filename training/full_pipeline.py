import json
import os
import optuna
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
from dataset_handler.frag import StyleDataModule, init_data_module
from models_handler.base.base_learner import BaseLearner
from models_handler.transformer.vit import VitClassifier
from training.optuna_hyper import just_a_wrapper
from utility.utility import BackboneType, HeadType
from torch.utils.data import DataLoader


def find_vit_hyper_and_train_model(
    dataset_path: str, 
    experiment_out_path: str,
    trial_per_head: int = 5,
    batch_size: int = 256,
    num_workers: int = 12,
    sampler: bool = False,
    use_masked_vit: bool = False, 
    use_countour: bool = False,
    head_type_esclusion: list[HeadType] = [HeadType.NONE, HeadType.SEQ_ENSEMBLE_CLS]
) -> BaseLearner:
    """
    Performs hyperparameter optimization and trains a Vision Transformer model.
    
    This function iterates through different classification head types, uses Optuna to find 
    the best hyperparameters for each head type, saves the results, and then trains a final 
    model using the best hyperparameters across all head types on the full dataset.
    
    Args:
        dataset_path (str): Path to the directory containing the dataset.
        experiment_out_path (str): Path where experiment results, checkpoints, and logs will be saved.
        trial_per_head (int, optional): Number of Optuna trials per head type. Defaults to 5.
        batch_size (int, optional): Batch size for data loading. Defaults to 256.
        num_workers (int, optional): Number of workers for data loading. Defaults to 12.
        sampler (bool, optional): Whether to use a custom sampler for data loading. Defaults to False.
        use_masked_vit (bool, optional): Whether to use masked Vision Transformer. Defaults to False.
        use_countour (bool, optional): Whether to use contour information. Defaults to False.
        head_type_esclusion (list[HeadType], optional): Head types to exclude from optimization. 
            Defaults to [HeadType.NONE, HeadType.SEQ_ENSEMBLE_CLS].
    
    Returns:
        BaseLearner: The trained Vision Transformer model with optimal hyperparameters.
    """
    
    best_loss: float | None = None
    best_trial: dict = {}
    best_head_type: str = ""
    
    data_module: StyleDataModule = init_data_module(
        data_dir=dataset_path,
        batch_size=batch_size, 
        num_workers=num_workers,
        sampler=sampler, 
        use_test=False, 
        use_masked_vit=use_masked_vit, 
        return_name=True, 
        use_contourn=use_countour
    )

    for headtype in HeadType:
        if headtype in head_type_esclusion:
            continue

        study = optuna.create_study(direction="minimize")  
        study.optimize(
            func=just_a_wrapper(
                model_type=BackboneType.VIT_16, 
                datamodule=data_module, 
                num_epoch=25, 
                contrastive_loss=False, 
                head_type=headtype,
                masked_attention=False, 
                backbone_class=VitClassifier, 
                out_dir=experiment_out_path
            ), 
            n_trials=trial_per_head, 
            n_jobs=1
        ) 
        print(f"Best hyperparameters for CLS ce loss with head_type: {headtype.name} --->", study.best_params)
        out_path = os.path.join(
            experiment_out_path, 
            f"best_hype_{headtype.name}_epoch_cls.json"
        )
        
        with open(out_path, "w") as f:
            json.dump(
                obj={
                    "data": study.best_params, 
                    "validation_loss": study.best_value
                }, 
                fp=f, 
                indent=4
            )

        if best_loss is None or best_loss > study.best_value:
            best_loss = study.best_value
            best_trial = study.best_params
            best_head_type = headtype.name

    backbone_type = BackboneType.VIT_16.name
    contrastive_loss = False
    full_dataset = True
    head_type = best_head_type
    k_classes = 11
    lr = best_trial["lr"]
    masked_attention = False
    min_epochs_head = best_trial["min_epochs_head"]
    use_weighted_loss = best_trial["weighted_loss"]
    weight_decay = best_trial["weight_decay"]

    data_module = init_data_module(
        data_dir=dataset_path,
        batch_size=batch_size, 
        num_workers=num_workers,
        sampler=sampler, 
        use_test=full_dataset, 
        use_masked_vit=use_masked_vit,
        use_contourn=use_countour
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

    base: str = os.path.join(experiment_out_path, "FINAL")
    # CSV logger
    logger_csv = CSVLogger(
        save_dir=os.path.join(base, "logs"),
        name=f"FINAL_VIT",
    )

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(base, f"chkt"),
        filename=f"weights",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )
    early_stopping_cb = pl.callbacks.EarlyStopping(
        monitor="val_loss",     # Metric to monitor
        mode="min",             # "min" for loss, "max" for accuracy/F1
        patience=8,             # Number of epochs with no improvement
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

    test_model(
        test_data_loader=data_module.test_dataloader(),
        test_result_path=os.path.join(base, "test"), 
        log_loss=True, 
        model=model
    )

    return model


def test_model(
    test_data_loader: DataLoader,
    test_result_path: str,
    log_loss: bool = False,
    model: BaseLearner | None = None,
    model_chkt_path: str | None = None,
    model_hparams_path: str | None = None
) -> None:
    """
    Evaluates a model on a test dataset.
    
    This function tests a model on the provided test data. It can either use a model instance 
    directly or load one from a checkpoint. Optionally, it logs the test loss and other metrics 
    to a CSV logger for later analysis.
    
    Args:
        test_data_loader (DataLoader): DataLoader for the test dataset.
        test_result_path (str): Path where test results will be saved.
        log_loss (bool, optional): Whether to log test loss and metrics to CSV. Defaults to False.
        model (BaseLearner | None, optional): An instantiated model to use for testing. 
            If None, the model will be loaded from checkpoint. Defaults to None.
        model_chkt_path (str | None, optional): Path to the model checkpoint file. 
            Required if model is None. Defaults to None.
        model_hparams_path (str | None, optional): Path to the model hyperparameters file. 
            Used when loading from checkpoint. Defaults to None.
    
    Returns:
        None
    """
    
    if model is None:
        model = BaseLearner.load_from_checkpoint(
            checkpoint_path=model_chkt_path,
            hparams_file=model_hparams_path
        )

    model.test_result_path = test_result_path

    trainer_param: dict = {
        "enable_progress_bar": True,
    }

    if log_loss:
        logger_csv = CSVLogger(
            save_dir=os.path.join(test_result_path, "logs"),
            name=f"report"
        )

        trainer_param["logger"] = logger_csv


    trainer = pl.Trainer(
        **trainer_param
    )

    trainer.test(
        model=model,
        dataloaders=test_data_loader
    )