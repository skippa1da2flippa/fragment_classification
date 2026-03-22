import json

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics as tm
import timm
from torch import Tensor
from dataset_handler.cleopatra_dist import get_dataset_weights
from timm.models.vision_transformer import VisionTransformer
from loss_function.supervised_contrastive_loss import SupConLoss
from models_handler.base.base_learner import BaseLearner
from utility.utility import BackboneType, CleopatraInput, CleopatraOut, HeadType
from timm.models.vision_transformer import global_pool_nlc


class VitClassifier(BaseLearner): #pl.LightningModule):
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
        backbone_type: str,
        head_type: str,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        min_epochs_head: int = 5,
        k_classes: int = 11,
        num_head_mha: int = 12, 
        use_weighted_loss: bool = False,
        contrastive_loss: bool = False, 
        masked_attention: bool = False, 
        full_dataset: bool = True
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Backbone (ViT or DeiT)
        self.backbone: VisionTransformer = timm.create_model(
            model_name=BackboneType[backbone_type].value,
            pretrained=True,
            global_pool=HeadType[head_type].value,
            num_classes=k_classes,
        )

        # At start up the model is frozen beside its head and its tail
        self.apply_params(
            value=False,
            module=self.backbone, 
            exclusion_lst=["head", "norm", "patch_embed"]
        )
        self.forzen_blocks_map: Tensor = torch.zeros(len(self.backbone.blocks))

        # Metrics — Lightning will handle aggregation & reset
        self.val_accuracy: tm.Metric = tm.Accuracy(task="multiclass", num_classes=k_classes) 
        self.val_f1: tm.Metric = tm.F1Score(task="multiclass", num_classes=k_classes) 
        self.val_auc: tm.Metric = tm.AUROC(task="multiclass", num_classes=k_classes) 

        # List of prediction for the test set
        self.test_distribution: list[Tensor] = []
        self.test_name: list[str] = []
        self.test_labels: list[Tensor] = []

        self._prev_val_loss: float = .0

        if use_weighted_loss:
            weights_tensor = get_dataset_weights(
                full_count=full_dataset
            ).float()  
        else:
            weights_tensor = torch.ones(size=(k_classes,), dtype=torch.float) 

        # This ensures it's always moved to the correct device with the model
        self.register_buffer("loss_weights", weights_tensor)

        if contrastive_loss:
            self.loss: nn.Module = SupConLoss()


    def apply_params(
        self, 
        module: nn.Module, 
        value: bool = False,
        exclusion_lst: list[str] = [], 
        use_block_map: bool = False
    ) -> None:
        
        if use_block_map:
            for flag, block in zip(self.forzen_blocks_map, module):
                self.apply_params(
                    module=block,
                    value=bool(flag)
                )
        else:
            for name, param in module.named_parameters():
                if not any(ex in name for ex in exclusion_lst):  # substring match
                    param.requires_grad = value


    def multi_task_forward(
        self,
        batch: Tensor, 
        attention_mask: Tensor = None, 
        aggregate: bool = True,
        norm: bool = True,
        dropout: bool = True,
        return_embedding: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        
        emb: Tensor = self.predict_embedding(
            batch=batch,
            attention_mask=attention_mask, 
            aggregate=aggregate,
            norm=norm,
            dropout=dropout, 
            return_all=return_embedding
        )
    
        data = emb[:, 0] if emb.dim() > 2 else emb
        logits: Tensor = self.backbone.head(data)

        if return_embedding: 
            return logits, emb
        
        else:
            return logits

    def forward(
        self, 
        batch: Tensor, 
        attention_mask: Tensor | None = None
    ) -> Tensor:
        
        # In this case the aggregation fun `global_pool_nlc()`
        # is called just onto the token which recieved
        # attention throughout the vit blocks
        if (
            self.hparams.masked_attention
        ):
            return self.multi_task_forward(
                batch=batch,
                attention_mask=attention_mask
            )

        if attention_mask is None:
            logits: Tensor = self.backbone(
                x=batch
            )
        else:
            # In this case the aggregation fun `global_pool_nlc()`
            # is called on the whole sequence not just
            # on the attended tokens I
            attention_mask = attention_mask.unsqueeze(dim=1)
            logits: Tensor = self.backbone(
                x=batch, 
                attn_mask=attention_mask
            )

        return logits

    def predict_step(self, batch: Tensor) -> Tensor:
        logits: Tensor = self(batch)
        prediction: Tensor = torch.argmax(
            input=logits, 
            dim=1
        )

        return prediction
    
    def predict_embedding(
        self, 
        batch: Tensor,
        attention_mask: Tensor | None = None,
        aggregate: bool = False,
        norm: bool = False,
        dropout: bool = False,
        return_all: bool = False
    ) -> Tensor:
        if attention_mask is None:
            out: Tensor =  self.backbone.forward_features(
                x=batch
            )

            attention_mask_diag: Tensor = torch.ones(
                out.shape[0],
                out.shape[1],
                dtype=out.dtype,
                device=out.device
            )
            
        else:
            attention_mask = attention_mask.unsqueeze(dim=1)
            patches: Tensor =  self.backbone.forward_features(
                x=batch,
                attn_mask=attention_mask
            )

            attention_mask_diag: Tensor = attention_mask.diagonal(dim1=1, dim2=2)
            out: Tensor = patches * attention_mask_diag

        if aggregate:
            pool_type: str = HeadType[self.hparams.head_type].value
            flag: bool = attention_mask is not None
            match pool_type, flag:
                case HeadType.SEQ_ENSEMBLE.value, True:
                    masked_mean: Tensor = out.sum(dim=1) / attention_mask_diag.sum(dim=1)
                    pooled_out: Tensor = masked_mean
                
                case  HeadType.SEQ_ENSEMBLE_MAX.value, True:
                    masked_mean: Tensor = out.sum(dim=1) / attention_mask_diag.sum(dim=1)
                    pooled_out: Tensor = 0.5 * (out.amax(dim=1) + masked_mean)

                case _:
                    pooled_out: Tensor = global_pool_nlc(
                        x=out,
                        pool_type=pool_type
                    )

            if return_all:
                # TODO you might need the cls
                out = torch.concat([
                        pooled_out.unsqueeze(dim=1), 
                        out[:, 1:, :]
                    ], 
                    dim=1
                )
            else:
                out = pooled_out

        if norm:
            out = self.backbone.fc_norm(
                out
            )
        
        if dropout:
            out = self.backbone.head_drop(out)

        return out
        
        
    def base_step(
            self, 
            batch: CleopatraInput, 
            step_type: str = "train"
        ) -> CleopatraOut:
        img, attention_mask, label = batch

        weights = self.loss_weights if step_type == "train" else torch.ones_like(self.loss_weights) 

        if self.hparams.masked_attention:
            logits: Tensor = self(
                batch=img, 
                attention_mask=attention_mask
            )
        else: 
            logits: Tensor = self(
                batch=img
            )
           
        if not self.hparams.contrastive_loss:
            loss: Tensor = F.cross_entropy(
                input=logits, 
                target=label, 
                weight=weights
            ) 
        else:
            logits = logits / logits.sum(dim=1, keepdim=True)
            loss: Tensor = self.loss(
                input=logits.unsqueeze(dim=1), 
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
        self.log(
            name="train_loss", 
            value=loss, 
            prog_bar=True, 
            on_step=True, 
            on_epoch=True,
            batch_size=batch.image.shape[0]
        )
        return loss


    def validation_step(self, batch: CleopatraInput, batch_idx: int) -> None:
        loss, logits, preds, labels = self.base_step(
            batch=batch, 
            step_type="val"
        )

        # log loss per batch → averaged by Lightning at epoch end
        self.log(
            name="val_loss", 
            value=loss, 
            prog_bar=True, 
            on_step=False, 
            on_epoch=True,
            batch_size=batch.image.shape[0]
        )

        if not self.hparams.contrastive_loss:
            # update metrics
            self.val_accuracy.update(preds, labels)
            self.val_f1.update(preds, labels)
            self.val_auc.update(logits, labels)

    def test_step(self, batch: CleopatraInput, batch_idx: int) -> None:
        loss, logits, preds, labels = self.base_step(
            batch=batch, 
            step_type="test"
        )

        # log loss per batch → averaged by Lightning at epoch end
        self.log(
            name="test_loss", 
            value=loss, 
            prog_bar=True, 
            on_step=False, 
            on_epoch=True, 
            batch_size=batch.image.shape[0]
        )

        self.test_distribution.append(torch.softmax(logits, dim=1))
        self.test_name.extend(batch[1]) # Assuming we have the name at postion one 
        self.test_labels.append(labels)

        
    def on_train_end(self) -> None:
        self.apply_params(
            value=False,
            module=self.backbone, 
            exclusion_lst=["head", "norm", "patch_embed"]
        )
        self.forzen_blocks_map.zero_()

    def on_validation_epoch_end(self) -> None:
        if not self.hparams.contrastive_loss:
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

        self.unfreezing_handler()

    def test_epoch_end(self) -> None:
        res: list[dict] = []
        
        lst_test_distribution: list[list[float]] = [ten.tolist() for batch in self.test_distribution for ten in batch]
        lst_test_labels: list[int] = [label.item() for batch in self.test_labels for label in batch]

        agg_coll = zip(
            self.test_name, 
            lst_test_distribution, 
            lst_test_labels
        )
        for name, prob, label in agg_coll:
            res.append(
                {
                    "name": name,
                    "prob": prob,
                    "label": label
                }
            )

        with open("test_vit.json", "w") as f:
            json.dump(res, f, indent=2)

        
    def unfreezing_handler(
        self, 
        val_loss_name: str = "val_loss", 
        log_blocks: bool = True,
        plateau_threshold: float = 1e-4,
        min_epoch: int | None = None
    ) -> None:

        if log_blocks:
            self.log_trainable_blocks()

        val_loss: float = self.trainer.callback_metrics[val_loss_name].item()

        # Perform progressive unfreezing only if the backbone.blocks is not 
        # totaly unfrozen
        if self.forzen_blocks_map.sum() < len(self.backbone.blocks):

            min_epoch = min_epoch if min_epoch else self.hparams.min_epochs_head
            # Minimum head-only epochs
            if self.current_epoch < min_epoch:
                self._prev_val_loss = val_loss
            else:
                if not any(p.requires_grad for p in self.backbone.head.parameters()):
                    for p in self.backbone.head.parameters():
                        p.requires_grad = True
                else:     
                    if val_loss >= self._prev_val_loss - plateau_threshold:
                        # unfreeze one block from head to tail
                        last_zero = (self.forzen_blocks_map == 0).nonzero()[-1]
                        self.forzen_blocks_map[last_zero] = 1
                        self.apply_params( 
                            module=self.backbone.blocks,
                            use_block_map=True
                        )

                    self._prev_val_loss = val_loss

    def configure_optimizers(self):
        backbone_decay, backbone_no_decay = [], []
        head_decay, head_no_decay = [], []

        for name, param in self.named_parameters():
            
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
            {"params": backbone_decay, "weight_decay": self.hparams.weight_decay, "lr": self.hparams.lr * 0.1},
            {"params": backbone_no_decay, "weight_decay": 0.0, "lr": self.hparams.lr * 0.1},
            {"params": head_decay, "weight_decay": self.hparams.weight_decay, "lr": self.hparams.lr},
            {"params": head_no_decay, "weight_decay": 0.0, "lr": self.hparams.lr},
        ])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


    def log_trainable_blocks(self) -> None:
        self.log(
            "trainable_blocks_count", 
            self.forzen_blocks_map.sum(), 
            prog_bar=False, 
            on_epoch=True
        )
        
