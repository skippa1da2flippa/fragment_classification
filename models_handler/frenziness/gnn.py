import torch
import torch.nn as nn
import pytorch_lightning as pl
from utility.utility import CleopatraInput, CleopatraOut, GNNType, GraphGenout, generate_connection
from torch import Tensor
import torch.nn.functional as F
# from dataset_handler.cleopatra_dist import weights
from torch_geometric.data import Batch
import torchmetrics as tm

class UltimateGraphApproach(pl.LightningModule):
    def __init__(
        self, 
        gnn_type: str, 
        gnn_num_layer: int,
        backbone_type: type[pl.LightningModule], 
        backbone_weight_path: str,
        backbone_hparam_path: str,
        initial_emb_size: int = 768,
        final_head_size: int = 11,
        gnn_act_fun: str = "relu",
        gnn_dropout: float = 0.2,
        graph_load_param: float = 0.7,
        use_weighted_loss: bool = False,
        brain_adaptation: bool = False,
        lr: float = 0.01,
        weight_decay: float = 0.003
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vit_embedder: pl.LightningModule = backbone_type.load_from_checkpoint(
            checkpoint_path=backbone_weight_path,
            hparams_file=backbone_hparam_path
        )

        self.gnn: nn.Module = GNNType[gnn_type].value(
            in_channels=initial_emb_size,
            hidden_channels=initial_emb_size,
            num_layers=gnn_num_layer,
            out_channels=final_head_size, 
            act=gnn_act_fun, 
            dropout=gnn_dropout
        )

        # if use_weighted_loss:
        #     weights_tensor: Tensor = weights.clone().float()  
        # else:
        #     weights_tensor: Tensor = torch.ones(
        #         size=(final_head_size,), 
        #         dtype=torch.float
        #     )

        # # This ensures it's always moved to the correct device with the model
        # self.register_buffer("loss_weights", weights_tensor)

         # Metrics — Lightning will handle aggregation & reset
        self.val_accuracy: tm.Metric = tm.Accuracy(task="multiclass", num_classes=final_head_size)
        self.val_f1: tm.Metric = tm.F1Score(task="multiclass", num_classes=final_head_size)
        self.val_auc: tm.Metric = tm.AUROC(task="multiclass", num_classes=final_head_size)
        
        self.vit_embedder.freeze()


    def configure_optimizers(self):
        gnn_decay, gnn_no_decay = [], []

        for name, param in self.gnn.named_parameters():
            if "bias" in name or "norm" in name.lower():
                gnn_no_decay.append(param)
            else:
                gnn_decay.append(param)

        optimizer = torch.optim.AdamW([
            {"params": gnn_decay, "weight_decay": self.hparams.weight_decay, "lr": self.hparams.lr},
            {"params": gnn_no_decay, "weight_decay": 0.0, "lr": self.hparams.lr},
        ])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def on_validation_epoch_end(self) -> None:
        metrics: dict[str, Tensor] = {
            "val_accuracy": self.val_accuracy.compute(),
            "val_f1": self.val_f1.compute(),
            "val_auc": self.val_auc.compute(),
        }
        self.log_dict(metrics, prog_bar=True)
        
        # Reset metrics
        self.val_accuracy.reset()
        self.val_f1.reset()
        self.val_auc.reset()

    def base_step(
            self, 
            batch: CleopatraInput, 
            step_type: str = "train"
        ) -> CleopatraOut:
        img, _, label = batch
        logits: Tensor = self(img)

        if step_type == "train":
            loss: Tensor = F.cross_entropy(
                input=logits, 
                target=label, 
                weight=self.loss_weights
            )
        else: 
            loss: Tensor = F.cross_entropy(
                input=logits, 
                target=label
            )

        preds: Tensor = torch.argmax(logits, dim=1)
        return CleopatraOut(
            loss=loss, 
            logits=logits, 
            prediction=preds, 
            label=label
        )


    def training_step(self, batch: CleopatraInput, batch_idx: int) -> Tensor:
        loss, _, _, _ = self.base_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss


    def validation_step(self, batch: CleopatraInput, batch_idx: int) -> None:
        loss, logits, preds, labels = self.base_step(
            batch=batch, 
            step_type="val"
        )

        # log loss per batch → averaged by Lightning at epoch end
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # update metrics
        self.val_accuracy.update(preds, labels)
        self.val_f1.update(preds, labels)
        self.val_auc.update(logits, labels)

            
    def forward(self, batch: Tensor) -> Tensor:
        patches: Tensor = self.vit_embedder.predict_embedding(batch)
        graph_out: GraphGenout = generate_connection(
            patches_emb=patches,
            load_param=self.hparams.graph_load_param, 
            device=self.device
        )

        graph_batch = Batch.from_data_list(graph_out.graph_batch).to(self.device)

        # Step 4: Forward through the GNN
        # Assuming your GNN internally handles pooling and returns logits per graph
        logits: Tensor = self.gnn(
            x=graph_batch.x,
            edge_index=graph_batch.edge_index, 
            batch=graph_batch.batch
        )

        # Return just the logits related to the cls vector of each graph
        return logits[0::patches.shape[1], :]

    def predict_step(self, batch: Tensor) -> Tensor:
        logits: Tensor = self(batch)
        prediction: Tensor = torch.argmax(
            input=logits, 
            dim=1
        )

        return prediction