from typing import Type
import torch
import torch.nn as nn
from torch import Tensor
from models_handler.base.base_ensemble import BaseEnsemble
from models_handler.base.base_learner import BaseLearner
from utility.utility import EnsembleForwardInput, EnsembleForwardOut

class Aggregator(nn.Module):

    def __init__(self, num_learners: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = nn.Parameter(torch.ones(num_learners) / num_learners)

    
    def forward(self, logits: Tensor) -> Tensor:
        # logits shape: (batch_size, num_learners, num_classes)
        # weights shape: (num_learners,)
        normalized_weights = torch.softmax(self.weights, dim=0)
        weighted_logits = logits * normalized_weights.view(1, -1, 1)
        return weighted_logits.sum(dim=1)


class BaselineEnsemble(BaseEnsemble):
    def __init__(
        self, 
        model_dataset_info: list[int],
        model_paths: list[tuple[str, str]], 
        model_types: list[Type[BaseLearner]] | Type[BaseLearner],
        learners_name: list[str] = [],
        learner_loss_regulizer: float = .5,
        final_head_size: int = 11, 
        use_weighted_loss: bool = False,
        full_dataset: bool = True,
        lr: float = 0.01,
        weight_decay: float = 0.003,
        mask_on_learner: int = 2,
    ) -> None:
        
        super().__init__(
            model_paths=model_paths,
            model_types=model_types,
            handler_model=Aggregator(num_learners=len(model_paths)),
            handler_model_name="aggregator",
            learners_name=learners_name,
            min_epoch_handler_model=1,
            final_head_size=final_head_size,
            use_weighted_loss=use_weighted_loss,
            full_dataset=full_dataset,
            lr=lr,
            weight_decay=weight_decay,
            mask_on_learner=mask_on_learner, 
            learner_loss_regulizer=learner_loss_regulizer
        )

        self.save_hyperparameters(
            {
                "model_dataset_info": model_dataset_info
            }
        )


    def forward(
        self, 
        batch: EnsembleForwardInput
    ) -> EnsembleForwardOut:
        
        learners_logits, _ = self.learners_forward(
            batch_lst=batch.batch_lst,
            attention_mask=batch.attention_mask, 
            return_tensor=True, 
            idx_lst=self.hparams.model_dataset_info
        )

        ensemble_logits: Tensor = self.handler_model(learners_logits)

        return EnsembleForwardOut(
            ensemble_logits=ensemble_logits, 
            learners_logits=learners_logits
        )
    



    