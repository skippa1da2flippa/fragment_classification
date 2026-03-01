import torch
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics as tm
import pytorch_lightning as pl
import timm
from torch import Tensor
from dataset_handler.cleopatra_dist import weights
from timm.models.vision_transformer import VisionTransformer
from loss_function.supervised_contrastive_loss import SupConLoss
from utility.utility import BackboneType, CleopatraInput, HeadType
from typing import Any, Dict, Tuple


class VitContrastive(pl.LightningModule):
    """ 
    PyTorch Lightning module for training a Vision Transformer (ViT) or DeiT-based classifier. 
    This class wraps a pretrained backbone (ViT-16 or DeiT-16, pretrained on ImageNet) and 
    allows customization of the optimization setup, fine-tuning pace, and classifier head.

    Parameters ----------
        backbone_type : str
            The backbone architecture type. Must correspond to a key in BackboneType 
            (e.g., "VIT_16" or "DEIT_16").
    
        lr : float
            Learning rate for the optimizer.

        weight_decay : float
            Weight decay factor applied during optimization.

        min_epochs_head : int
            The minimum number of epochs for training only the head (before unfreezing the backbone).

        head_type : str
            Classifier head type. Must correspond to a key in HeadType (e.g., "CLS_SINGLE" or "SEQ_ENSEMBLE").

        k_classes : int, default=11
            Number of output classes for classification. Default is 11, but can be adjusted depending on your dataset.
    """

    def __init__(
        self,
        backbone: pl.LightningModule,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        k_classes: int = 11,
        emb_in: int = 768,
        use_weighted_loss: bool = False
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['backbone'])

        # Backbone (ViT or DeiT)
        self.backbone: VisionTransformer = backbone
        self.backbone.freeze()

        # At start up the model is frozen beside its head and its tail
        self.head: nn.Linear = nn.Linear(
            in_features=emb_in, 
            out_features=k_classes
        )

        # Metrics — Lightning will handle aggregation & reset
        self.val_accuracy: tm.Metric = tm.Accuracy(task="multiclass", num_classes=k_classes)
        self.val_f1: tm.Metric = tm.F1Score(task="multiclass", num_classes=k_classes)
        self.val_auc: tm.Metric = tm.AUROC(task="multiclass", num_classes=k_classes)

        self._prev_val_loss: float = .0

        if use_weighted_loss:
            weights_tensor = weights.clone().float()  
        else:
            weights_tensor = torch.ones(size=(k_classes,), dtype=torch.float)

        # This ensures it's always moved to the correct device with the model
        self.register_buffer("loss_weights", weights_tensor)
                

    def forward(self, batch: Tensor) -> Tensor:
        patches: Tensor =  self.backbone.backbone.forward_features(batch)
        patches = patches[:, 1:, :].mean(dim=1)
        return self.head(patches)

    def predict_step(self, batch: Tensor) -> Tensor:
        logits: Tensor = self(batch)
        prediction: Tensor = torch.argmax(
            input=logits, 
            dim=1
        )

        return prediction

    def base_step(
            self, 
            batch: CleopatraInput, 
            step_type: str = "train"
        ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
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
        return loss, logits, preds, label


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

        self.val_accuracy.update(preds, labels)
        self.val_f1.update(preds, labels)
        self.val_auc.update(logits, labels)

    def on_validation_epoch_end(self) -> None:
      
        metrics: Dict[str, Tensor] = {
            "val_accuracy": self.val_accuracy.compute(),
            "val_f1": self.val_f1.compute(),
            "val_auc": self.val_auc.compute(),
        }
        self.log_dict(metrics, prog_bar=True)
        
        # Reset metrics
        self.val_accuracy.reset()
        self.val_f1.reset()
        self.val_auc.reset()


    def configure_optimizers(self):
        backbone_decay, backbone_no_decay = [], []
        head_decay, head_no_decay = [], []

        for name, param in self.named_parameters():
            if param.requires_grad:
                if "head" in name:  # classifier head
                    if "bias" in name or "norm" in name.lower():
                        head_no_decay.append(param)
                    else:
                        head_decay.append(param)
                else:  # backbone
                    if "bias" in name or "norm" in name.lower():
                        backbone_no_decay.append(param)
                    else:
                        backbone_decay.append(param)

        optimizer = torch.optim.AdamW([
            {"params": head_decay, "weight_decay": self.hparams.weight_decay, "lr": self.hparams.lr},
            {"params": head_no_decay, "weight_decay": 0.0, "lr": self.hparams.lr},
        ])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

