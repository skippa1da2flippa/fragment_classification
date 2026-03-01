from torch import Tensor
import torch
import torch.nn as nn
from loss_function.kl_sup_con_loss import KL_ContrastiveLoss
from utility.utility import CleopatraInput, CleopatraOut, HeadType
from models_handler.transformer.vit import VitClassifier
import torch.nn.functional as F
import math


class KlVIT(VitClassifier):
    def __init__(
        self,
        backbone_type: str,
        head_type: str,
        double_head: bool = True,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        min_epochs_head: int = 5,
        k_classes: int = 11,
        num_head_mha: int = 12, 
        use_weighted_loss: bool = False,
        contrastive_loss: bool = False, 
        masked_attention: bool = False,
        alpha: float  = 1.,
        beta: float | None = None,
        kl_reduction: str = "sum",
        kl_symmetric: bool = True,
        ce_minimum_epoch: int = 3,
        temperature: float = 6.,
        p_plus: bool = False
    ) -> None:
        
        super().__init__(
            backbone_type=backbone_type,
            head_type=head_type,
            lr=lr,
            weight_decay=weight_decay,
            min_epochs_head=min_epochs_head,
            k_classes=k_classes,
            num_head_mha=num_head_mha,
            use_weighted_loss=use_weighted_loss,
            contrastive_loss=contrastive_loss,
            masked_attention=masked_attention,
        )

        if double_head:
            self.kl_head: nn.Module = nn.Linear(
                in_features=768,
                out_features=k_classes
            )

        self.save_hyperparameters()

        self.kl_loss: nn.Module = KL_ContrastiveLoss(
            symmetric=kl_symmetric, 
            reduction=kl_reduction, 
            temperature=temperature,
            p_plus=p_plus
        )

    def double_head_handler(
        self, 
        batch: CleopatraInput
    ) -> tuple[Tensor, Tensor]:
        img, _, _ = batch
        res: Tensor = super().predict_embedding(img)

        match self.hparams.head_type:
            case HeadType.CLS_SINGLE.name:
                res = res[:, 0, :]

            case HeadType.SEQ_ENSEMBLE.name:
                res = res[:, 1:, :].mean(dim=1)

            case HeadType.SEQ_ENSEMBLE_CLS.name:
                res = res.mean(dim=1)

        pre_logits: Tensor = self.backbone.norm(res)
        ce_logits: Tensor = self.backbone.head(pre_logits)

        if self.current_epoch >= self.hparams.ce_minimum_epoch:
            kl_logits: Tensor = self.kl_head(pre_logits)
        else:
            kl_logits: Tensor = self.kl_head(pre_logits.detach())

        return ce_logits, kl_logits

    def base_step(
        self, 
        batch: CleopatraInput, 
        step_type: str = "train"
    ) -> CleopatraOut:

        img, _, label = batch

        weights = self.loss_weights if step_type == "train" else torch.ones_like(self.loss_weights) 
        
        if self.hparams.double_head:
            ce_logits, kl_logits = self.double_head_handler(
                batch=batch
            )

        else:
            ce_logits = kl_logits = self.__call__(img)

        kl_loss: Tensor = self.kl_loss(
            input=kl_logits, 
            target=label
        )
        
        ce_loss: Tensor = F.cross_entropy(
            input=ce_logits, 
            target=label, 
            weight=weights
        ) 

        preds: Tensor = torch.argmax(ce_logits, dim=1)

        self.log(f"kl_{step_type}_loss", kl_loss, prog_bar=True, on_step=False, on_epoch=True)
        if (
            self.current_epoch < self.hparams.ce_minimum_epoch 
            and not self.hparams.double_head
        ):
            beta: float = .0
        else:
            beta: float = self.hparams.beta

        self.log(f"kl_beta", beta, prog_bar=True, on_step=False, on_epoch=True)

        return CleopatraOut(
            loss=self.hparams.alpha * ce_loss + beta * kl_loss, 
            logits=ce_logits, 
            prediction=preds, 
            label=label
        )
    

    # def on_train_epoch_end(self) -> None:
    #     max_epochs: int = self.trainer.max_epochs
    #     beta_start: float = 0.0  # optional starting beta
    #     beta_end: float = self.hparams.beta if self.hparams.beta is not None else 1.0
    #     self.hparams.beta = beta_start + 0.5 * (beta_end - beta_start) * (1 - math.cos(math.pi * self.current_epoch / max_epochs))

    #     self.log(f"kl_beta", self.hparams.beta, prog_bar=True, on_step=False, on_epoch=True)