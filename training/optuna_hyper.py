import os
from typing import Callable
import optuna as op 
import pytorch_lightning as pl
from models_handler.base.base_ensemble import BaseEnsemble
from models_handler.ensemble.graph_ensemble import GraphEnsemble
from models_handler.ensemble.weighted_avg_ensemble import WeightedAverageEnsemble
from models_handler.frenziness.gnn import UltimateGraphApproach
from utility.utility import BackboneType, GNNType, HeadType
from models_handler.transformer.vit import VitClassifier
from pytorch_lightning.loggers import CSVLogger

def just_a_wrapper(
    model_type: BackboneType, 
    head_type: HeadType,
    datamodule: pl.LightningDataModule, 
    out_dir: str,
    backbone_class: type[pl.LightningModule] = VitClassifier,
    num_epoch: int = 40, 
    k_classes: int = 11, 
    contrastive_loss: bool = False, 
    masked_attention: bool = False,
    ultimate_loss: bool = False, 
    device: str ="cuda"
) -> Callable[[op.trial.Trial], float]:

    def objective(trial: op.trial.Trial) -> float:
        # Hyperparameter search space
        backbone_type = model_type.name # trial.suggest_categorical("backbone_type", ["VIT_16", "DEIT_16"])
        lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        min_epochs_head = trial.suggest_int("min_epochs_head", 1, 10)
        # trial.suggest_categorical("head_type", ["CLS_SINGLE", "SEQ_ENSEMBLE"])
        alpha = None
        beta = None
        kl_symmetric = True
        kl_reduction = "sum"
        k_classes = 11
        
        if contrastive_loss:
            use_weighted_loss: bool = trial.suggest_categorical("weighted_loss", [False])
            k_classes = trial.suggest_categorical("dim_out", [50, 100, 150])
    
        else:
            use_weighted_loss: bool = trial.suggest_categorical("weighted_loss", [True, False])

        # Create the model with the current set of hyperparameters
        model = backbone_class(
            backbone_type=backbone_type,
            lr=lr,
            weight_decay=weight_decay,
            min_epochs_head=min_epochs_head,
            head_type=head_type.name,
            k_classes=k_classes, 
            use_weighted_loss=use_weighted_loss, 
            contrastive_loss=contrastive_loss, 
            masked_attention=masked_attention
        )

        # CSV logger
        logger_csv = CSVLogger(
            save_dir=os.path.join(f"{out_dir}", f"logs_{head_type.name}"),
            name=f"csv",
        )

        check_point_saver = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(f"{out_dir}", f"checkpoints_{head_type.name}"), 
            filename=f"weights", 
            monitor='val_loss', 
            mode='min'
        )
        early_stopping_cb = pl.callbacks.EarlyStopping(
            monitor="val_loss",     # Metric to monitor
            mode="min",             # "min" for loss, "max" for accuracy/F1
            patience=5,             # Number of epochs with no improvement
            min_delta=1e-3,         # Required improvement threshold
            verbose=True
        )

        trainer = pl.Trainer(
            max_epochs=num_epoch,
            logger=logger_csv,
            callbacks=[
                check_point_saver,
                early_stopping_cb
            ],
            enable_progress_bar=True,
            accelerator=device
        )

        trainer.fit(
            model=model, 
            datamodule=datamodule
        )

        return check_point_saver.best_model_score.item()
    
    return objective


def ultimate_graph_wrapper(
    datamodule: pl.LightningDataModule,
    backbone_type: type[pl.LightningModule],
    backbone_weight_path: str,
    backbone_hparam_path: str,
    num_epoch: int = 30,
    final_head_size: int = 11,
    gnn_name: str = ""
) -> callable:
    """
    Returns an Optuna objective function for UltimateGraphApproach hyperparameter tuning.
    """

    def objective(trial: op.trial.Trial) -> float:
        
        if gnn_name != "":
            gnn_candidates = [gnn_name]
        else:
            gnn_candidates = [e.name for e in GNNType]

        gnn_type = trial.suggest_categorical(
            "gnn_type", gnn_candidates
        )
        gnn_num_layer = trial.suggest_int("gnn_num_layer", 1, 3)
        gnn_act_fun = trial.suggest_categorical("gnn_act_fun", ["relu", "gelu", "elu"])
        gnn_dropout = trial.suggest_float("gnn_dropout", 0.0, 0.6)
        graph_load_param = trial.suggest_float("graph_load_param", 0.1, 1)
        use_weighted_loss = trial.suggest_categorical("use_weighted_loss", [True, False])

        # Optimizer params
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        # -----------------------------
        # 🧠 Create model
        # -----------------------------
        model = UltimateGraphApproach(
            gnn_type=gnn_type,
            gnn_num_layer=gnn_num_layer,
            backbone_type=backbone_type,
            backbone_weight_path=backbone_weight_path,
            backbone_hparam_path=backbone_hparam_path,
            initial_emb_size=768,
            final_head_size=final_head_size,
            gnn_act_fun=gnn_act_fun,
            gnn_dropout=gnn_dropout,
            graph_load_param=graph_load_param,
            use_weighted_loss=use_weighted_loss,
            weight_decay=weight_decay,
            lr=lr
        )

        # -----------------------------
        # 🧾 Logger + Checkpoint
        # -----------------------------
        base_path: str = f"optuna_{gnn_type}_{graph_load_param}_{gnn_num_layer}_{gnn_act_fun}"
        logger_csv = CSVLogger(
            save_dir=os.path.join(base_path, f"optuna_logs_{gnn_type}"),
            name=f"{gnn_type}_trial_{trial.number}"
        )

        checkpoint_cb = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(base_path, f"optuna_ckpt/{gnn_type}"),
            filename=f"trial_{trial.number}",
            monitor="val_loss",
            mode="min",
            save_top_k=1
        )
        early_stopping_cb = pl.callbacks.EarlyStopping(
            monitor="val_loss",     # Metric to monitor
            mode="min",             # "min" for loss, "max" for accuracy/F1
            patience=5,             # Number of epochs with no improvement
            min_delta=1e-4,         # Required improvement threshold
            verbose=True
        )

        # -----------------------------
        # 🚀 Train
        # -----------------------------
        trainer = pl.Trainer(
            max_epochs=num_epoch,
            logger=logger_csv,
            callbacks=[checkpoint_cb, early_stopping_cb],
            enable_progress_bar=False,
            accelerator="auto",
            devices=1
        )

        trainer.fit(model=model, datamodule=datamodule)

        # -----------------------------
        # 🎯 Return metric to optimize
        # -----------------------------
        # You can also return negative accuracy/F1 if you prefer maximizing
        score = checkpoint_cb.best_model_score.item() if checkpoint_cb.best_model_score else float("inf")
        return score

    return objective


def the_chosen(
    model_type: BackboneType, 
    datamodule: pl.LightningDataModule, 
    model_params: dict,
    out_dir: str,
    backbone_class: type[pl.LightningModule] = VitClassifier,
    p_plus: bool = False,
    num_epoch: int = 40, 
    k_classes: int = 11,
    device: str ="cuda"
) -> Callable[[op.trial.Trial], float]:

    def objective(trial: op.trial.Trial) -> float:
        # Hyperparameter search space
        backbone_type = model_type.name # trial.suggest_categorical("backbone_type", ["VIT_16", "DEIT_16"])
        lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        
        use_weighted_loss: bool = trial.suggest_categorical("weighted_loss", [True, False])
        alpha = trial.suggest_categorical("alpha_loss", [0.5, 1, 1.5, 2, 3])
        beta = trial.suggest_categorical("beta_loss", [0.5, 1, 1.5, 2, 3])
        kl_symmetric = trial.suggest_categorical("kl_symmetric", [True, False])
        kl_reduction = trial.suggest_categorical("kl_reduction", ["mean", "sum"]) 
        ce_minimum_epoch = trial.suggest_categorical("ce_minimum_epoch", [1, 2, 3, 4]) 
        temperature = trial.suggest_categorical("kl_temperature", [6., 7, 7.5, 8, 9, 9.5, 10])

        body = {
            "backbone_type": backbone_type,
            "lr": lr,
            "weight_decay": weight_decay,
            "k_classes": k_classes, 
            "use_weighted_loss": use_weighted_loss, 
            "alpha": alpha, 
            "beta": beta,
            "kl_symmetric": kl_symmetric,
            "kl_reduction": kl_reduction,
            "ce_minimum_epoch": ce_minimum_epoch,
            "temperature": temperature,
            "p_plus": p_plus,
            **model_params
        }

        # Create the model with the current set of hyperparameters
        model = backbone_class(
            **body
        )

        head_type = model_params["head_type"]

        # CSV logger
        logger_csv = CSVLogger(
            save_dir=f"{out_dir}\\logs_{head_type}",
            name=f"csv",
        )

        check_point_saver = pl.callbacks.ModelCheckpoint(
            dirpath=f'{out_dir}\\checkpoints_{head_type}', 
            filename=f"weights", 
            monitor='val_loss', 
            mode='min'
        )
        early_stopping_cb = pl.callbacks.EarlyStopping(
            monitor="kl_val_loss",     # Metric to monitor
            mode="min",             # "min" for loss, "max" for accuracy/F1
            patience=5,             # Number of epochs with no improvement
            min_delta=1e-3,         # Required improvement threshold
            verbose=True
        )

        trainer = pl.Trainer(
            max_epochs=num_epoch,
            logger=logger_csv,
            callbacks=[
                check_point_saver,
                early_stopping_cb
            ],
            enable_progress_bar=True,
            accelerator=device
        )

        trainer.fit(
            model=model, 
            datamodule=datamodule
        )

        return check_point_saver.best_model_score.item()
    
    return objective


def ensemble_graph_wrapper(
    datamodule: pl.LightningDataModule,
    model_paths: list[tuple[str, str]], 
    model_types: type[BaseEnsemble],
    decision_mode: str,
    bs_path: str,
    num_epoch: int = 30,
    gnn_name: str = ""
) -> callable:
    """
    Returns an Optuna objective function for UltimateGraphApproach hyperparameter tuning.
    """

    def objective(trial: op.trial.Trial) -> float:

        model_dataset_info = [0, 1, 0]
        learners_name=[
            "base_vit", 
            "extr_vit",
            "mskd_vit"
        ]
        
        if gnn_name != "":
            gnn_candidates = [gnn_name]
        else:
            gnn_candidates = [e.name for e in GNNType]

        gnn_type = trial.suggest_categorical(
            "gnn_type", gnn_candidates
        )
        gnn_num_layer = trial.suggest_int("gnn_num_layer", 1, 3)
        gnn_act_fun = trial.suggest_categorical("gnn_act_fun", ["relu", "gelu", "elu", "tanh"])
        gnn_dropout = trial.suggest_float("gnn_dropout", 0.0, 0.6)
        graph_load_param = trial.suggest_float("graph_load_param", 0.1, 1)
        use_weighted_loss = trial.suggest_categorical("use_weighted_loss", [True, False])
        min_epoch_gnn = 1 # trial.suggest_int("min_epoch_gnn", 1, 7)
        central_node_mode = trial.suggest_categorical("central_node_mode", ["mean", "zero"])
        temperature = trial.suggest_float("temperature", 0.5, 9)
        keep_temperature_stable = trial.suggest_categorical("keep_temperature_stable", [True, False])

        # Optimizer params
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        # -----------------------------
        # 🧠 Create model
        # -----------------------------
        model = GraphEnsemble(
            model_paths=model_paths,
            model_types=model_types,
            learners_name=learners_name, 
            gnn_type=gnn_type, 
            model_dataset_info=model_dataset_info, 
            min_epoch_gnn=min_epoch_gnn, 
            gnn_num_layer=gnn_num_layer,
            gnn_dropout=gnn_dropout,
            gnn_act_fun=gnn_act_fun,
            use_weighted_loss=use_weighted_loss,
            decision_mode=decision_mode,
            central_node_mode=central_node_mode,
            temperature=temperature,
            keep_temperature_stable=keep_temperature_stable, 
            lr=lr,
            weight_decay=weight_decay, 
            graph_load_param=graph_load_param
        )

        # -----------------------------
        # 🧾 Logger + Checkpoint
        # -----------------------------
        base_path: str = os.path.join(bs_path, f"{gnn_type}")
        logger_csv = CSVLogger(
            save_dir=os.path.join(base_path), 
            name=decision_mode
        )

        checkpoint_cb = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(base_path, f"ckpt"),
            filename=f"{decision_mode}",
            monitor=f"{gnn_type}_val_loss",
            mode="min",
            save_top_k=1
        )
        early_stopping_cb = pl.callbacks.EarlyStopping(
            monitor=f"{gnn_type}_val_loss",     # Metric to monitor
            mode="min",             # "min" for loss, "max" for accuracy/F1
            patience=5,             # Number of epochs with no improvement
            min_delta=1e-4,         # Required improvement threshold
            verbose=False
        )

        # -----------------------------
        # 🚀 Train
        # -----------------------------
        trainer = pl.Trainer(
            max_epochs=num_epoch,
            logger=logger_csv,
            callbacks=[checkpoint_cb, early_stopping_cb],
            enable_progress_bar=True,
            accelerator="auto",
            devices=1
        )

        trainer.fit(model=model, datamodule=datamodule)

        # -----------------------------
        # 🎯 Return metric to optimize
        # -----------------------------
        # You can also return negative accuracy/F1 if you prefer maximizing
        score = checkpoint_cb.best_model_score.item() if checkpoint_cb.best_model_score else float("inf")
        return score

    return objective


def ensemble_weighted_wrapper(
    datamodule: pl.LightningDataModule,
    model_paths: list[tuple[str, str]], 
    model_types: type[BaseEnsemble],
    bs_path: str,
    num_epoch: int = 30
) -> callable:
    """
    Returns an Optuna objective function for UltimateGraphApproach hyperparameter tuning.
    """

    def objective(trial: op.trial.Trial) -> float:

        model_dataset_info = [0, 1, 0]
        learners_name=[
            "base_vit", 
            "extr_vit",
            "mskd_vit"
        ]
        
        mlp_num_layer = trial.suggest_int("mlp_num_layer", 1, 3)
        mlp_act_fun = trial.suggest_categorical("mlp_act_fun", ["RELU", "GELU", "TANH", "SIGMOID"])
        mlp_dropout = trial.suggest_float("mlp_dropout", 0.0, 0.6)
        use_weighted_loss = trial.suggest_categorical("use_weighted_loss", [True, False])
        min_epoch_mlp = trial.suggest_int("min_epoch_mlp", 1, 7)
        temp_regulator = trial.suggest_loguniform("temperature_regulator", 0.005, 0.01)

        # Optimizer params
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        # -----------------------------
        # 🧠 Create model
        # -----------------------------
        model = WeightedAverageEnsemble(
            model_paths=model_paths,
            model_types=model_types,
            learners_name=learners_name, 
            model_dataset_info=model_dataset_info, 
            min_epoch_mlp=min_epoch_mlp, 
            mlp_num_layer=mlp_num_layer,
            mlp_dropout=mlp_dropout,
            mlp_act_fun=mlp_act_fun,
            use_weighted_loss=use_weighted_loss,
            lr=lr,
            weight_decay=weight_decay, 
            temp_reg=temp_regulator
        )

        # -----------------------------
        # 🧾 Logger + Checkpoint
        # -----------------------------
        base_path: str = os.path.join(bs_path, f"{mlp_act_fun}")
        logger_csv = CSVLogger(
            save_dir=os.path.join(base_path, f"logs"),
            name=f"log"
        )

        checkpoint_cb = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(base_path, f"ckpt"),
            filename=f"trial_{trial.number}",
            monitor=f"mlp_val_loss",
            mode="min",
            save_top_k=1
        )
        early_stopping_cb = pl.callbacks.EarlyStopping(
            monitor=f"mlp_val_loss",     # Metric to monitor
            mode="min",             # "min" for loss, "max" for accuracy/F1
            patience=5,             # Number of epochs with no improvement
            min_delta=1e-4,         # Required improvement threshold
            verbose=False
        )

        # -----------------------------
        # 🚀 Train
        # -----------------------------
        trainer = pl.Trainer(
            max_epochs=num_epoch,
            logger=logger_csv,
            callbacks=[checkpoint_cb, early_stopping_cb],
            enable_progress_bar=True,
            accelerator="auto",
            devices=1
        )

        trainer.fit(model=model, datamodule=datamodule)

        # -----------------------------
        # 🎯 Return metric to optimize
        # -----------------------------
        # You can also return negative accuracy/F1 if you prefer maximizing
        score = checkpoint_cb.best_model_score.item() if checkpoint_cb.best_model_score else float("inf")
        return score

    return objective