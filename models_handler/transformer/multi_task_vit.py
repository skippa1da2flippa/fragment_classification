from models_handler.transformer.vit import VitClassifier
from torch import Tensor, ones_like, stack
from typing import Callable
import torch.nn as nn
import torch.nn.functional as F
from utility.utility import CleopatraEnsembleInput, CleopatraMultitaskOut, CleopatraOut

class MultiTaskVit(VitClassifier):
    """Vision Transformer with additional task heads and fusion loss."""
    def __init__(
        self, 
        backbone_type: str, 
        head_type: str, 
        lr: float, 
        weight_decay: float, 
        min_epochs_head: int, 
        out_dim_add_task: list[int],
        losses_weight: Tensor,
        label_getter: Callable[[Tensor], Tensor],
        n_add_task: int = 1,
        k_classes: int = 11, 
        use_weighted_loss: bool = False, 
        contrastive_loss: bool = False, 
        masked_attention: bool = False,
    ) -> None:
        
        super().__init__(
            backbone_type=backbone_type, 
            head_type=head_type, 
            lr=lr, 
            weight_decay=weight_decay, 
            min_epochs_head=min_epochs_head, 
            k_classes=k_classes, 
            use_weighted_loss=use_weighted_loss, 
            contrastive_loss=contrastive_loss, 
            masked_attention=masked_attention
        )

        self.label_getter: Callable[[Tensor], Tensor] = label_getter

        self.save_hyperparameters(
            {
                "n_add_task": n_add_task
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
        weights: Tensor,
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
                target=self.label_getter(label), 
                weight=weights
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
            attention_mask=attention_mask
        )

        final_loss: Tensor = self.losses_fusion(
            logits=logits, 
            label=label, 
            step_type=step_type
        )
        preds: Tensor = self.logits_fusion(logits=logits)    
    
        return CleopatraOut(
            loss=final_loss, 
            logits=logits, 
            prediction=preds, 
            label=label
        )
    
    def predict_step(self, batch: Tensor) -> Tensor:
        logits: Tensor = self(batch)
        prediction: Tensor = self.logits_fusion(logits=logits)

        return prediction

    def logits_fusion(self, logits: CleopatraMultitaskOut) -> Tensor:
        # TODO sara create the logic to fuse the logits of the different tasks
        pass