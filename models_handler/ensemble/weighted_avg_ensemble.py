import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Type
from models_handler.base.base_ensemble import BaseEnsemble
from models_handler.base.base_learner import BaseLearner
from utility.utility import ActFunEnum, CleopatraEnsembleInput, EnsembleForwardOut
import torch.nn as nn
from torchrl.modules import MLP
from torch.optim import Optimizer

class WeightedAverageEnsemble(BaseEnsemble):
    def __init__(
        self, 
        model_paths: list[tuple[str, str]], 
        model_types: list[Type[BaseLearner]] | Type[BaseLearner],
        model_dataset_info: list[int],
        mlp_num_layer: int,
        learners_name: list[str] = [],
        learner_loss_regulizer: float = 0.2,
        min_epoch_mlp: int = 5,
        temp_reg: float = .005,
        initial_emb_size: int = 768,
        final_head_size: int = 11,
        mlp_act_fun: str = "RELU",
        mlp_dropout: float = 0.2,
        use_weighted_loss: bool = False,
        full_dataset: bool = True,
        lr: float = 0.01,
        weight_decay: float = 0.003,
        mask_on_learner: int = 2
    ) -> None:
        
        mlp: nn.Module = MLP(
            in_features=initial_emb_size,
            out_features=1,
            depth=mlp_num_layer,
            activation_class=ActFunEnum[mlp_act_fun].value,
            dropout=mlp_dropout
        )
        
        super().__init__(
            model_paths=model_paths,
            model_types=model_types,
            handler_model=mlp,
            handler_model_name="mlp",
            learners_name=learners_name,
            min_epoch_handler_model=min_epoch_mlp,
            final_head_size=final_head_size,
            use_weighted_loss=use_weighted_loss,
            full_dataset=full_dataset,
            lr=lr,
            weight_decay=weight_decay,
            mask_on_learner=mask_on_learner, 
            learner_loss_regulizer=learner_loss_regulizer
        )

        self.save_hyperparameters({
            "mlp_act_fun": mlp_act_fun,
            "mlp_dropout": mlp_dropout,
            "mlp_num_layer": mlp_num_layer,
            "model_dataset_info": model_dataset_info,
            "temperature_loss_regulator": temp_reg
        })

        self.log_temperature: nn.Parameter = nn.Parameter(
            data=torch.zeros(
                len(model_paths), 
                requires_grad=True
            )
        ) 


    def forward(
        self, 
        batch_lst: list[Tensor], 
        attention_mask: Tensor | None = None
    ) -> EnsembleForwardOut:
        
        learners_logits, learners_embedding = self.learners_forward(
            batch_lst=batch_lst,
            attention_mask=attention_mask, 
            return_tensor=True, 
            idx_lst=self.hparams.model_dataset_info
        )

        importance_weight: Tensor = self.handler_model(learners_embedding[:, :, 0])
        norm_importance_weight: Tensor = torch.softmax(importance_weight, dim=1) 
        
        temperature: Tensor = torch.exp(self.log_temperature).view(1, self.log_temperature.shape[0], 1)
        scaled_learner_logits: Tensor = learners_logits / temperature

        ensemble_logits: Tensor = (norm_importance_weight * scaled_learner_logits).sum(dim=1)

        return EnsembleForwardOut(
           ensemble_logits=ensemble_logits,
           learners_logits=learners_logits
        )
    

    # def training_step(self, batch: CleopatraEnsembleInput, batch_idx: int) -> Tensor:
    #     ensemble_loss: Tensor = super().training_step(
    #         batch=batch, 
    #         batch_idx=batch_idx
    #     )

    #     temperature: Tensor = self.log_temperature.exp()      
    #     temperature = ((temperature - 1.0) ** 2).mean()

    #     loss: Tensor = ensemble_loss + self.hparams.temp_reg * temperature
    #     return loss


    def on_fit_start(self) -> None:
        for learner in self.learners:
            learner.trainer = self.trainer

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        optimizer, data = super().configure_optimizers()

        optimizer[0].add_param_group(
            param_group={
                "params": [self.log_temperature],
                "lr": self.hparams.lr,
                "weight_decay": 0.0
            }
        )

        return optimizer, data
    