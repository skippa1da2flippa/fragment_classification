import pytorch_lightning as pl
from torch import Tensor
import torch
import torch.nn.functional as F
from timm.models.vision_transformer import global_pool_nlc
from models_handler.transformer.gnn_vision_transformer import GraphVisionTransformer
from models_handler.transformer.vit import VitClassifier
from utility.utility import CleopatraOut, GraphVitInput, HeadType


class GraphVit(VitClassifier):
    

    def __init__(
        self,
        backbone_type: str,
        head_type: str,
        gnn_type: str,
        gnn_num_layer: int = 1,
        bpt_percentage: float = 0.9,
        cosine_threshold: float = 0.7,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        min_epoch: int = 5,
        k_classes: int = 11,
        num_head_mha: int = 12, 
        use_weighted_loss: bool = False,
        contrastive_loss: bool = False, 
        masked_attention: bool = False, 
        full_dataset: bool = True, 
        db_path: str = ""         
    ) -> None:
        
        super().__init__(
            backbone_type=backbone_type,
            head_type=head_type,
            lr=lr,
            weight_decay=weight_decay,
            min_epochs_head=min_epoch,
            k_classes=k_classes,
            num_head_mha=num_head_mha,
            use_weighted_loss=use_weighted_loss,
            contrastive_loss=contrastive_loss,
            masked_attention=masked_attention,
            full_dataset = full_dataset, 
            db_path= db_path
        )

        self.save_hyperparameters(
            {
                "gnn_type": gnn_type, 
                "bpt_percentage": bpt_percentage,
                "cosine_threshold": cosine_threshold,
                "gnn_num_layer": gnn_num_layer
            }
        )

        self._build_model()



    def _build_model(self) -> None:

        self.backbone = GraphVisionTransformer.build_from_vision_transformer(
            vit_model=self.backbone,    
            gnn_type=self.hparams.gnn_type,
            gnn_num_layer=self.hparams.gnn_num_layer
        )

    def on_train_start(self) -> None:
        torch.autograd.set_detect_anomaly(True)

        # TODO see if these two are necessary
        # technically they shouldn't
        self.backbone.head.requires_grad = True
        self.backbone.norm.requires_grad = True

        for block in self.backbone.blocks:
            for name, param in block.named_parameters():
                if "graph_local_attn" in name or "norm0" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False


    def predict_embedding(
        self, 
        batch: Tensor,
        attention_mask: Tensor | None = None,
        bpt_partitions: Tensor | None = None,
        aggregate: bool = False,
        norm: bool = False,
        dropout: bool = False,
        return_all: bool = False
    ) -> Tensor:
        
        if attention_mask is None:
            out: Tensor = self.backbone.forward_features(
                x=batch, 
                bpt_partitions=bpt_partitions
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
                attn_mask=attention_mask,
                bpt_partitions=bpt_partitions
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


    def multi_task_forward(
        self,
        batch: Tensor, 
        attention_mask: Tensor = None, 
        bpt_partitions: Tensor | None = None,
        aggregate: bool = True,
        norm: bool = True,
        dropout: bool = True,
        return_embedding: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        
        emb: Tensor = self.predict_embedding(
            batch=batch,
            attention_mask=attention_mask, 
            bpt_partitions=bpt_partitions,
            aggregate=aggregate,
            norm=norm,
            dropout=dropout, 
            return_all=return_embedding
        )

        global_token: Tensor = emb[:, 0] if emb.dim() > 2 else emb
        logits: Tensor = self.backbone.head(global_token)

        if return_embedding: 
            return logits, emb 
        else:
            return logits



    def forward(
        self,
        batch: Tensor, 
        attention_mask: Tensor | None = None,
        bpt_partitions: Tensor | None = None
    ) -> Tensor:
        
        if self.hparams.masked_attention:
            logits: Tensor = self.multi_task_forward(
                batch=batch,
                attention_mask=attention_mask, 
                bpt_partitions=bpt_partitions
            )

        else: #if attention_mask is None:
            logits: Tensor = self.backbone(
                x=batch, 
                bpt_partitions=bpt_partitions
            )

        return logits

    def base_step(
        self, 
        batch: GraphVitInput, 
        step_type: str = "train"
    ) -> CleopatraOut:
        
        if isinstance(batch, GraphVitInput):
            img, label, attention_mask, bpt_info, _ = batch
        else:
            img, label, attention_mask, _ = batch
            bpt_info = None

        weights = self.loss_weights if step_type == "train" else torch.ones_like(self.loss_weights) 

        logits: Tensor = self(
            batch=img, 
            attention_mask=attention_mask, 
            bpt_partitions=bpt_info
        )
        
        loss: Tensor = F.cross_entropy(
            input=logits, 
            target=label, 
            weight=weights
        ) 


        preds: Tensor = torch.argmax(logits, dim=1)
        return CleopatraOut(
            loss=loss, 
            logits=logits, 
            prediction=preds, 
            label=label
        )
    
    def configure_optimizers(self):
        backbone_decay, backbone_no_decay = [], []
        head_decay, head_no_decay = [], []
        gnn_decay, gnn_no_decay = [], []

        for name, param in self.named_parameters():
            
            if "head" in name:  # classifier head
                if "bias" in name or "norm" in name.lower():
                    head_no_decay.append(param)
                else:
                    head_decay.append(param)
                    
            elif "graph_local_attn" in name or "norm0" in name:  # local attention and its norm
                if "bias" in name or "norm" in name.lower():
                    gnn_no_decay.append(param)
                else:
                    gnn_decay.append(param)
            else:  # backbone
                if "bias" in name or "norm" in name.lower():
                    backbone_no_decay.append(param)
                else:
                    backbone_decay.append(param)

        optimizer = torch.optim.AdamW([
            {"params": backbone_decay, "weight_decay": self.hparams.weight_decay, "lr": self.hparams.lr * 0.1},
            {"params": backbone_no_decay, "weight_decay": 0.0, "lr": self.hparams.lr * 0.1},
            {"params": gnn_decay, "weight_decay": self.hparams.weight_decay, "lr": self.hparams.lr},
            {"params": gnn_no_decay, "weight_decay": 0.0, "lr": self.hparams.lr},
            {"params": head_decay, "weight_decay": self.hparams.weight_decay, "lr": self.hparams.lr},
            {"params": head_no_decay, "weight_decay": 0.0, "lr": self.hparams.lr},
        ])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]








    