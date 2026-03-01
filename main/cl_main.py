from dataset_handler.frag import init_data_module
from pytorch_lightning.loggers import CSVLogger
from models_handler.frenziness.gnn import UltimateGraphApproach
from models_handler.transformer.vit import VitClassifier
import pytorch_lightning as pl
import os 

if __name__ == '__main__':
    data_module = init_data_module(
        data_dir="dataset",
        batch_size=20, 
        num_workers=12,
        sampler=False, 
        use_test=True
    )
    
    # Create the model with the current set of hyperparameters
    model = VitClassifier(
        backbone_type="VIT_16",
        lr=0.0001584460127273449,
        weight_decay=4.037216149183175e-05,
        min_epochs_head=3,
        head_type="CLS_SINGLE",
        k_classes=11, 
        use_weighted_loss=False, 
        contrastive_loss=False
    )

    base = "final_VIT"
    # CSV logger
    logger_csv = CSVLogger(
        save_dir=f"final_VIT\\FULL_VIT_TEST_logs",
        name=f"FINAL_VIT_csv",
    )

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(base, f"FINAL_VIT_CHKT"),
        filename=f"weights",
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

    # brain_adaptation = False
    # final_head_size = 11
    # gnn_act_fun = "gelu"
    # gnn_dropout = 0.5025478950176492
    # gnn_num_layer = 3
    # gnn_type = "GAT"
    # graph_load_param = 0.32475031628950934
    # initial_emb_size = 768
    # lr = 9.951536636170876e-05
    # use_weighted_loss = False
    # weight_decay = 0.0003114189827151807


    # print(f"RUNNING GAT WITH {gnn_num_layer}-LAYERS, AND ALPHA: {graph_load_param} ...")

    # model = UltimateGraphApproach(
    #         gnn_type=gnn_type,
    #         gnn_num_layer=gnn_num_layer,
    #         backbone_type=VitClassifier,
    #         backbone_weight_path="final_VIT\\FINAL_VIT_CHKT\\weights_extrapolated.ckpt",
    #         backbone_hparam_path="final_VIT\\FULL_VIT_TEST_logs\\FINAL_VIT_csv_extrapolated\\version_0\\hparams_extrapolated.yaml",
    #         initial_emb_size=768,
    #         final_head_size=11,
    #         gnn_act_fun=gnn_act_fun,
    #         gnn_dropout=gnn_dropout,
    #         graph_load_param=graph_load_param,
    #         use_weighted_loss=use_weighted_loss,
    #         weight_decay=weight_decay,
    #         lr=lr
    #     )

    # # -----------------------------
    # # 🧾 Logger + Checkpoint
    # # -----------------------------
    # base_path: str = f"FINAL_GNN_INNER_{gnn_type}"
    # logger_csv = CSVLogger(
    #     save_dir=base_path,
    #     name=f"{gnn_type}"
    # )

    # checkpoint_cb = pl.callbacks.ModelCheckpoint(
    #     dirpath=os.path.join(base_path, f"{gnn_type}_wght"),
    #     filename=f"weights",
    #     monitor="val_loss",
    #     mode="min",
    #     save_top_k=1
    # )
    # early_stopping_cb = pl.callbacks.EarlyStopping(
    #     monitor="val_loss",     # Metric to monitor
    #     mode="min",             # "min" for loss, "max" for accuracy/F1
    #     patience=5,             # Number of epochs with no improvement
    #     min_delta=1e-4,         # Required improvement threshold
    #     verbose=False
    # )

    # # -----------------------------
    # # 🚀 Train
    # # -----------------------------
    # trainer = pl.Trainer(
    #     max_epochs=30,
    #     logger=logger_csv,
    #     callbacks=[
    #         checkpoint_cb,
    #         early_stopping_cb
    #     ],
    #     enable_progress_bar=True,
    #     accelerator="auto",
    #     devices=1
    # )

    # trainer.fit(model=model, datamodule=data_module)