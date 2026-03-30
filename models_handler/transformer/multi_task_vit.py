import torch
from models_handler.transformer.vit import VitClassifier
from torch import Tensor, ones_like, stack
from typing import Callable
import torch.nn as nn
import torch.nn.functional as F
from utility.utility import CleopatraEnsembleInput, CleopatraMultitaskOut, CleopatraOut, get_epoch_per_style

class MultiTaskVit(VitClassifier):
    """Vision Transformer with additional task heads and fusion loss."""
    def __init__(
        self, 
        backbone_type: str, 
        head_type: str, 
        lr: float, 
        weight_decay: float, 
        min_epochs_head: int, 
        losses_weight: Tensor,
        label_getter: Callable[[Tensor], Tensor] | None = None, # SARA: input style_id output is epoch_id
        out_dim_add_task: list[int] = [3],
        n_add_task: int = 1,
        k_classes: int = 11, 
        use_weighted_loss: bool = False, 
        masked_attention: bool = False,
        fusion_params: Tensor | None = None,
    ) -> None:
        
        super().__init__(
            backbone_type=backbone_type, 
            head_type=head_type, 
            lr=lr, 
            weight_decay=weight_decay, 
            min_epochs_head=min_epochs_head, 
            k_classes=k_classes, 
            use_weighted_loss=use_weighted_loss, 
            masked_attention=masked_attention
        )

        if label_getter is None:
            self.label_getter: Callable[[Tensor], Tensor] = get_epoch_per_style
        else:
            self.label_getter: Callable[[Tensor], Tensor] = label_getter

        self.save_hyperparameters(
            {
                "n_add_task": n_add_task,
            }
        )

        self.heads: nn.ModuleList = nn.ModuleList()

        for task_id in range(n_add_task):
            self.heads.append(
                nn.Linear(
                    in_features=self.backbone.head.in_features,
                    out_features=out_dim_add_task[task_id]
                )
            )

        # This ensures it's always moved to the correct device with the model
        self.register_buffer("multi_task_loss_weights", losses_weight)
        self.register_buffer("fusion_params", fusion_params)
        
    def forward(
        self, 
        batch: Tensor, 
        attention_mask: Tensor | None = None
    ) -> CleopatraMultitaskOut:
        """Forward pass for multi-task inference.

        Computes the main style logits via the inherited ViT pipeline, then applies
        each auxiliary head to the pooled token embedding to produce additional
        task logits. Returns all logits in a `CleopatraMultitaskOut` payload.
        """

        all_logits: list[Tensor] = []

        style_logits, token = self.multi_task_forward(
            batch=batch, 
            attention_mask=attention_mask, 
            aggregate=True, 
            norm=True, 
            dropout=True, 
            return_embedding=True
        )

        all_logits.append(style_logits)

        for head in self.heads:
            head_logits: Tensor = head(token[:, 0])
            all_logits.append(head_logits)

        return CleopatraMultitaskOut(
            logits=all_logits,
        )
    
    def losses_fusion(
        self, 
        logits: CleopatraMultitaskOut, 
        label: Tensor,
        step_type: str = "train"
    ) -> Tensor:
        """Fuse style + auxiliary losses into a single scalar.

        For the main style logit (logits.logits[0]) and each auxiliary head logit,
        computes cross entropy against the appropriate label targets. The class may
        use train-mode weights for the style loss (self.loss_weights), else uniform
        weights (in evaluation mode). The final loss is computed via the
        registered `multi_task_loss_weights` weighted average.
        """

        multi_task_loss: list[Tensor] = []
        weights = self.loss_weights if step_type == "train" else ones_like(self.loss_weights) 

        multi_task_loss.append(
            F.cross_entropy(
                input=logits.logits[0], 
                target=label, 
                weight=weights
            )
        )
        
        for head_logits in logits.logits[1:]:
            loss: Tensor = F.cross_entropy(
                input=head_logits, 
                target=self.label_getter(label)
            ) 

            multi_task_loss.append(loss)

        final_loss: Tensor = self.multi_task_loss_weights @ stack(multi_task_loss)

        return final_loss
    

    def base_step(
        self, 
        batch: CleopatraEnsembleInput, 
        step_type: str = "train"
    ) -> CleopatraOut:
        """Compute loss and predictions for a single iteration.

        Used by the training/validation loop to run forward pass, calculate
        multi-task loss, and produce fused prediction logits packaged in
        `CleopatraOut` for downstream metric logging.
        """

        img, label, attention_mask, _ = batch
        
        logits: CleopatraMultitaskOut = self(
            batch=img, 
            attention_mask=attention_mask if self.hparams.masked_attention else None
        )

        final_loss: Tensor = self.losses_fusion(
            logits=logits, 
            label=label, 
            step_type=step_type
        )
        preds: Tensor = self.logits_fusion(logits=logits)    
    
        return CleopatraOut(
            loss=final_loss, 
            logits=logits.logits[0], 
            prediction=preds, 
            label=label
        )
    
    def predict_step(self, batch: Tensor) -> Tensor:
        logits: Tensor = self(batch)
        prediction: Tensor = self.logits_fusion(logits=logits)

        return prediction

    # def logits_fusion(self, logits: CleopatraMultitaskOut) -> Tensor:
    #     """Return fused prediction based on epoch confidence and style logits.

    #     Strategy:
    #     - Take the epoch prediction from the first auxiliary head (logits.logits[1]).
    #     - Compute style softmax over logits.logits[0].
    #     - For each sample, select the highest-probability style that belongs to the
    #       predicted epoch (using `label_getter` as style -> epoch mapper).
    #     - If the currently best style already belongs to that epoch, keep it.

    #     This implements: "if model is more sure about 1st epoch then to the 1st style
    #     to find first style that belongs to that epoch".
    #     """
    #     style_logits = logits.logits[0]
    #     epoch_logits = logits.logits[1]

    #     # Predicted epoch for each sample
    #     predicted_epoch = epoch_logits.argmax(dim=1)

    #     # Style probabilities and predicted style
    #     style_probs = F.softmax(style_logits, dim=1)
    #     predicted_style = style_probs.argmax(dim=1)

    #     # Map each style index to its epoch via provided label_getter
    #     style_idx = torch.arange(style_logits.size(1), device=style_logits.device)
    #     style_to_epoch = self.label_getter(style_idx)

    #     # If predicted style already belongs to predicted epoch, keep it
    #     matched_style_epoch = style_to_epoch[predicted_style]
    #     keep_current = matched_style_epoch == predicted_epoch

    #     # For samples where style and epoch disagree, find best style under predicted epoch
    #     epoch_mask = predicted_epoch.unsqueeze(1) == style_to_epoch.unsqueeze(0)

    #     # Mask invalid styles to -inf so they are not selected by argmax.
    #     safe_probs = style_probs.clone()
    #     safe_probs[~epoch_mask] = float("-inf")
    #     epoch_selected_style = safe_probs.argmax(dim=1)

    #     final_style = torch.where(keep_current, predicted_style, epoch_selected_style)

    #     return final_style

    def logits_fusion(self, logits: CleopatraMultitaskOut) -> Tensor:
        """Fused prediction using weighted sum of style and epoch logits."""
        style_logits = logits.logits[0] 
        epoch_logits = logits.logits[1]

        # Use hparams for the weights and divisor
        w_s = self.fusion_params[0]
        w_e = self.fusion_params[1]
        div = self.fusion_params[2]

        # 1. Map style indices to epoch indices
        style_indices = torch.arange(style_logits.size(1), device=style_logits.device)
        style_to_epoch = self.label_getter(style_indices).long()

        # 2. Align epoch logits to the 11 style dimensions
        # This handles the "unordered" mapping correctly
        expanded_epoch_logits = epoch_logits[:, style_to_epoch]

        # 3. Weighted Fusion
        fused_logits = (w_s * style_logits + w_e * expanded_epoch_logits) / div

        return fused_logits.argmax(dim=1) 
    
    def configure_optimizers(self):
        backbone_decay, backbone_no_decay = [], []
        head_decay, head_no_decay = [], []
        heads_decay, heads_no_decay = [], []

        for name, param in self.named_parameters():
            if "heads" in name:  # additional task heads
                if "bias" in name or "norm" in name.lower():
                    heads_no_decay.append(param)
                else:
                    heads_decay.append(param)
            elif "head" in name:  # classifier head
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
            {"params": heads_decay, "weight_decay": self.hparams.weight_decay, "lr": self.hparams.lr},
            {"params": heads_no_decay, "weight_decay": 0.0, "lr": self.hparams.lr},
        ])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]