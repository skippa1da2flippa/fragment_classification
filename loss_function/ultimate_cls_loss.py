import torch
import torch.nn as nn
from torch import Tensor 
from loss_function.kl_sup_con_loss import KL_ContrastiveLoss


class UltimateClsLoss(nn.Module):
    def __init__(
        self, 
        alpha: float,
        beta: float,
        ce_reduction: str = "mean", 
        kl_reduction: str = "sum", 
        kl_symmetric: bool = True,
        weight: Tensor| None = None,
        n_classes: int = 11
    ) -> None:
        super().__init__()
        
        self.weight = weight if weight is not None else torch.ones((n_classes, ))
        self.ce: nn.Module = nn.CrossEntropyLoss(
            reduction=ce_reduction,
            weight=self.weight
        )

        self.kl: nn.Module = KL_ContrastiveLoss(
            symmetric=kl_symmetric, 
            reduction=kl_reduction
        )

        self.alpha: float = alpha
        self.beta: float = beta

    
    def forward(
        self, 
        input: Tensor, 
        target: Tensor
    ) -> Tensor:
        ce: Tensor = self.ce(
            input=input, 
            target=target,
        )

        kl: Tensor = self.kl(
            input=input, 
            target=target,
            weight=self.weight.to(input.device)
        )

        return (
            self.alpha * ce 
            +
            self.beta * kl
        )

