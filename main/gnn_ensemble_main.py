import json

import optuna
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
from dataset_handler.frag import init_data_module_ensemble
from models_handler.ensemble.graph_ensemble import GraphEnsemble
from models_handler.transformer.vit import VitClassifier
import os

from training.optuna_hyper import ensemble_graph_wrapper 


if __name__ == "__main__":

    mask_on_learner: int = 2

    data_module = init_data_module_ensemble(
        data_dirs=[
            "dataset", 
            "extrapolated_dataset"
        ], 
        num_workers=5, 
        batch_size=80, 
        use_test=True
    )

    model_paths = [
        (
            "final_VIT\\FINAL_VIT_CHKT\\weights.ckpt", 
            "final_VIT\\FULL_VIT_TEST_logs\\FINAL_VIT_csv\\version_0\\hparams.yaml"
        ), 
        (
            "final_VIT\\FINAL_VIT_CHKT\\weights_extrapolated-v2.ckpt",
            "final_VIT\\FULL_VIT_TEST_logs\\FINAL_VIT_csv_extrapolated\\version_4\\hparams.yaml"
        ),
        (
            "final_VIT\\FINAL_VIT_CHKT\\weights_masked_head_upd_wo_CLS.ckpt",
            "final_VIT\\FULL_VIT_TEST_logs\\FINAL_VIT_csv_masked_head_upd_wo_CLS\\version_1\\hparams.yaml"
        )
    ]


    # for w in [0.6, 0.7, 0.8, 0.9]:
    #     model = GraphEnsemble(
    #         model_paths=model_paths,
    #         model_types=VitClassifier,
    #         learners_name=[
    #             "base_vit", 
    #             "extr_vit",
    #             "mskd_vit"
    #         ], 
    #         gnn_type="GAT", 
    #         model_dataset_info=[0, 1, 0], 
    #         min_epoch_gnn=5, 
    #         gnn_num_layer=2, 
    #         temperature=0.6611976761416533, 
    #         decision_mode="all", 
    #         graph_load_param=0.5711808045563314, 
    #         lr=9.688888515970183e-05, 
    #         weight_decay=7.4384267353129015e-06, 
    #         learner_loss_regulizer=w,
    #         keep_temperature_stable=True,
    #         gnn_dropout=0.5311684635834418,
    #         gnn_act_fun="gelu", 
    #         central_node_mode="zero"

    #     )

    #     base = "Graph_ENSEMBLE_NEW"
    #     # CSV logger
    #     logger_csv = CSVLogger(
    #         save_dir=os.path.join(base, str(w), "Graph_ENSEMBLE_logs"),
    #         name=f"Graph_ENSEMBLE_ALL",
    #     )

    #     checkpoint_cb = pl.callbacks.ModelCheckpoint(
    #         dirpath=os.path.join(base, str(w), "Graph_ENSEMBLE_CHKT"),
    #         filename=f"Graph_ENSEMBLE_ALL",
    #         monitor="GAT_val_loss",
    #         mode="min",
    #         save_top_k=1
    #     )
        # early_stopping_cb = pl.callbacks.EarlyStopping(
        #     monitor="GAT_val_loss", # Metric to monitor
        #     mode="min",             # "min" for loss, "max" for accuracy/F1
        #     patience=5,             # Number of epochs with no improvement
        #     min_delta=1e-4,         # Required improvement threshold
        #     verbose=False
        # )

        # -----------------------------
        # 🚀 Train
        # -----------------------------

    model = GraphEnsemble(
        model_paths=model_paths,
        model_types=VitClassifier,
        learners_name=[
            "base_vit", 
            "extr_vit",
            "mskd_vit"
        ], 
        gnn_type="GAT", 
        model_dataset_info=[0, 1, 0], 
        min_epoch_gnn=5, 
        gnn_num_layer=2, 
        temperature=0.6611976761416533, 
        decision_mode="least", 
        graph_load_param=0.5711808045563314, 
        lr=9.688888515970183e-05, 
        weight_decay=7.4384267353129015e-06, 
        learner_loss_regulizer=0.6,
        keep_temperature_stable=True,
        gnn_dropout=0.5311684635834418,
        gnn_act_fun="gelu", 
        central_node_mode="zero",
        mask_on_learner=mask_on_learner
    )

    base = "Graph_ENSEMBLE_FINAL"
    # CSV logger
    logger_csv = CSVLogger(
        save_dir=os.path.join(base, "Graph_ENSEMBLE_logs"),
        name=f"Graph",
    )

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(base, "Graph_ENSEMBLE_CHKT"),
        filename=f"Graph_ENSEMBLE",
        monitor="GAT_val_loss",
        mode="min",
        save_top_k=1
    )
    trainer = pl.Trainer(
        max_epochs=50,
        logger=logger_csv,
        callbacks=[checkpoint_cb], #early_stopping_cb],
        enable_progress_bar=True,
        accelerator="auto",
        devices=1
    )

    trainer.fit(
        model=model, 
        datamodule=data_module
    )

    base_path ="gnn_ensemble_experiment"

    # for name in ["GAT", "GRAPHSAGE"]: 
    #     for mod in ["least", "most"]:
    #         study = optuna.create_study(direction="minimize")  
    #         study.optimize(
    #             func=ensemble_graph_wrapper(
    #                 datamodule=data_module, 
    #                 model_paths=model_paths,
    #                 model_types=VitClassifier,
    #                 num_epoch=20, 
    #                 gnn_name=name, 
    #                 bs_path=base_path, 
    #                 decision_mode=mod
    #             ), 
    #             n_trials=5, 
    #             n_jobs=1
    #         ) 
    #         print(f"Best hyperparameters for Graph: {name} --->", study.best_params)
    #         out_path = os.path.join(
    #             base_path, 
    #             name,
    #             mod
    #         )

    #         os.makedirs(out_path, exist_ok=True)
    #         out_path = os.path.join(
    #             base_path, 
    #             name,
    #             mod, 
    #             f"best.json"
    #         )
            
    #         with open(out_path, "w") as f:
    #             json.dump(
    #                 obj={
    #                     "data": study.best_params, 
    #                     "test_loss": study.best_value
    #                 }, 
    #                 fp=f, 
    #                 indent=4
    #             )


    # data_module = init_data_module_ensemble(
    #     data_dirs=[
    #         "dataset", 
    #         "extrapolated_dataset"
    #     ], 
    #     num_workers=5, 
    #     batch_size=10, 
    #     use_test=True
    # )

    # for name in ["GAT", "GRAPHSAGE"]: 
    #     for mod in ["all"]:
    #         study = optuna.create_study(direction="minimize")  
    #         study.optimize(
    #             func=ensemble_graph_wrapper(
    #                 datamodule=data_module, 
    #                 model_paths=model_paths,
    #                 model_types=VitClassifier,
    #                 num_epoch=20, 
    #                 gnn_name=name, 
    #                 bs_path=base_path, 
    #                 decision_mode=mod
    #             ), 
    #             n_trials=5, 
    #             n_jobs=1
    #         ) 
    #         print(f"Best hyperparameters for Graph: {name} --->", study.best_params)
    #         out_path = os.path.join(
    #             base_path, 
    #             name,
    #             mod
    #         )

    #         os.makedirs(out_path, exist_ok=True)
    #         out_path = os.path.join(
    #             base_path, 
    #             name,
    #             mod, 
    #             f"best.json"
    #         )
            
    #         with open(out_path, "w") as f:
    #             json.dump(
    #                 obj={
    #                     "data": study.best_params, 
    #                     "test_loss": study.best_value
    #                 }, 
    #                 fp=f, 
    #                 indent=4
    #             )


