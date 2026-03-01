import torch.nn as nn
from torch import Tensor
import torch
from utility.utility import kl_similarity

class KL_ContrastiveLoss(nn.Module):
    def __init__(
        self, 
        symmetric: bool = True,
        reduction: str = "sum", 
        temperature: float = 6.,
        p_plus: bool = False
    ) -> None:
        super().__init__()

        self.symmetric: bool = symmetric
        self.reduction: str = reduction
        self.temperature: float = temperature
        self.p_plus: bool = p_plus

    def forward(
        self, 
        input: Tensor, 
        target: Tensor,
        weight: Tensor | None = None
    ) -> Tensor:
        kl_sim: Tensor = kl_similarity(
            logits=input,
            symmetric=self.symmetric,
            reduction=self.reduction, 
            weight=weight, 
            temperature=self.temperature
        )

        mask_neg_samples: Tensor = 1 - torch.eye(
            kl_sim.shape[0], 
            kl_sim.shape[0], 
            dtype=torch.float, 
            device=kl_sim.device
        )

        mask_pos_samples: Tensor = self.p_plus_handler(
            target=target,
            mask_neg_samples=mask_neg_samples,
            logits=input
        )

        neg_samples: Tensor = (
            kl_sim * mask_neg_samples
        ).sum(dim=-1, keepdim=True)

        pos_samples: Tensor = (
            kl_sim * mask_pos_samples
        )

        log_neg_samples: Tensor = torch.log(neg_samples)
        log_pos_samples: Tensor = pos_samples
        log_pos_samples[pos_samples != .0] = torch.log(pos_samples[pos_samples != .0])

        log_value: Tensor = (
            log_pos_samples 
            - log_neg_samples
        )

        n_sample: Tensor = mask_pos_samples.sum(dim=-1, keepdim=True)
        n_sample[n_sample == .0] += 1

        losses: Tensor = -torch.div(
            input=log_value * mask_pos_samples, 
            other=n_sample
        )

        return losses.sum(dim=[0, 1]) / (mask_pos_samples.sum(dim=[0, 1]) + 1e-6)
    

    def  p_plus_handler(
        self,
        target: Tensor, 
        mask_neg_samples: Tensor, 
        logits: Tensor
    ) -> Tensor:
    
        if self.p_plus:
            pred: Tensor = torch.argmax(
                input=logits, 
                dim=1
            )

            p_plus_mask: Tensor = (
                pred == target
            ).float()

            p_plus_mask = p_plus_mask.unsqueeze(dim=0)

        else: 
            p_plus_mask: Tensor = torch.ones_like(
                input=target
            ).unsqueeze(dim=0)

            
        mask_pos_samples: Tensor = torch.eq(
            input=target.unsqueeze(dim=1), 
            other=target.unsqueeze(dim=0)
        ) * mask_neg_samples

        return mask_pos_samples * p_plus_mask