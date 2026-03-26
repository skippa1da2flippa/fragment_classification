from abc import abstractmethod
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Type
import torchmetrics as tm
from dataset_handler.cleopatra_dist import get_dataset_weights
from models_handler.base.base_learner import BaseLearner
from utility.utility import CleopatraEnsembleInput, CleopatraOut, EnsembleForwardOut, LearnerForwardOut, make_metrics
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Optimizer


class BaseEnsemble(LightningModule):
    def __init__(
        self, 
        model_paths: list[tuple[str, str]], 
        handler_model: nn.Module,
        handler_model_name: str,
        model_types: list[Type[BaseLearner]] | Type[BaseLearner],
        learners_name: list[str] = [],
        learner_loss_regulizer: float = .5,
        min_epoch_handler_model: int = 5,
        final_head_size: int = 11, 
        use_weighted_loss: bool = False,
        full_dataset: bool = True,
        lr: float = 0.01,
        weight_decay: float = 0.003,
        mask_on_learner: int = 2,
    ) -> None:
        
        super().__init__()
        self.save_hyperparameters(ignore=["handler_model", "model_paths"])

        self.handler_model: nn.Module = handler_model

        if not isinstance(model_types, list):
            model_types = [model_types]

        if len(model_types) == 1:
            model_types = model_types * len(model_paths)

        self.learners_name: list[str] = []
        self.learners: nn.ModuleList[BaseLearner] = nn.ModuleList()
        for idx, (model_ty, model_param) in enumerate(zip(model_types, model_paths)):
            backbone_weight_path, backbone_hparam_path = model_param
            model: BaseLearner = model_ty.load_from_checkpoint(
                checkpoint_path=backbone_weight_path,
                hparams_file=backbone_hparam_path
            )
            model.freeze()

            self.learners.append(
                model
            )
 
            self.learners_name.append(
                model.__class__.__name__ + f"_{idx}"
            )



        self.learners_name = self.learners_name if not len(learners_name) else learners_name

        if use_weighted_loss:
            weights_tensor: Tensor = get_dataset_weights(
                full_count=full_dataset
            )
        else:
            weights_tensor: Tensor = torch.ones(
                size=(final_head_size,), 
                dtype=torch.float
            )

        # This ensures it's always moved to the correct device with the model
        self.register_buffer("loss_weights", weights_tensor)

        # one MetricCollection per model
        self.val_metrics = torch.nn.ModuleList(
            [make_metrics(final_head_size) for _ in range(len(self.learners) + 1)]
        )

        

    # TODO the flaw here is that you could add later the parameter yet to unfreeze 
    def learners_configure_optimizers(self) -> tuple[list[list[nn.Parameter]], list[list[nn.Parameter]]]:
        final_learner_decay, final_learner_no_decay = [], []

        for model in self.learners: 
            learner_decay, learner_no_decay = [], []
            for name, param in model.named_parameters():
                if "bias" in name or "norm" in name.lower():
                    learner_no_decay.append(param)
                else:
                    learner_decay.append(param)

            final_learner_decay.append(learner_decay)
            final_learner_no_decay.append(learner_no_decay)

        return final_learner_decay, final_learner_no_decay
    
    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict[str, Any]]]: 
        handler_model_decay, handler_model_no_decay = [], []
        learners_decay, learners_no_decay = self.learners_configure_optimizers()

        for name, param in self.handler_model.named_parameters():
            if "bias" in name or "norm" in name.lower():
                handler_model_no_decay.append(param)
            else:
                handler_model_decay.append(param)

        optimizer_lst: list[dict] = []

        for idx, param_blk_decay in enumerate(learners_decay):
            optimizer_lst.append(
                {
                    "params": param_blk_decay, 
                    "weight_decay": self.learners[idx].hparams.weight_decay, 
                    "lr": self.learners[idx].hparams.lr * 0.1
                } 
            )

        for idx, param_blk_no_decay in enumerate(learners_no_decay):
            optimizer_lst.append(
                {
                    "params": param_blk_no_decay, 
                    "weight_decay": .0, 
                    "lr": self.learners[idx].hparams.lr * 0.1
                }
            )

        optimizer_lst.extend([
            {"params": handler_model_decay, "weight_decay": self.hparams.weight_decay, "lr": self.hparams.lr},
            {"params": handler_model_no_decay, "weight_decay": .0, "lr": self.hparams.lr}
        ])

        optimizer = torch.optim.Adam(optimizer_lst)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
    
    def learners_forward(
        self, 
        batch_lst: list[Tensor], 
        idx_lst: list[int],
        attention_mask: Tensor | None = None, 
        return_tensor: bool = False
    ) -> LearnerForwardOut:
        
        learners_logits: list[Tensor] = []
        learners_embedding: list[Tensor] = []

        model_attention_mask: Tensor | None = None

        for idx, model in enumerate(self.learners):

            batch: Tensor = batch_lst[idx_lst[idx]]

            if self.hparams.mask_on_learner == idx:
                model_attention_mask = attention_mask
            else:
                model_attention_mask = None
                
            logits, embedding = model.multi_task_forward(
                batch=batch, 
                attention_mask=model_attention_mask, 
                aggregate=True,
                norm=True,
                dropout=True,
                return_embedding=True
            )

            learners_logits.append(logits)
            learners_embedding.append(embedding)

        if not return_tensor:
            out: LearnerForwardOut = LearnerForwardOut(
                learners_logits=learners_logits,
                learners_embedding=learners_embedding
            )
        else:
            learners_logits = [en.unsqueeze(dim=1) for en in learners_logits]
            learners_patches = [en.unsqueeze(dim=1) for en in learners_embedding]

            learners_logits_t: Tensor = torch.cat(learners_logits, dim=1)
            learners_patches_t: Tensor = torch.cat(learners_patches, dim=1)

            out: LearnerForwardOut = LearnerForwardOut(
                learners_logits=learners_logits_t,
                learners_embedding=learners_patches_t
            )

        return out

    

    @abstractmethod
    def forward(
        self, 
        batch_lst: list[Tensor], 
        attention_mask: Tensor | None = None
    ) -> EnsembleForwardOut:
        pass
    
    
    def base_step(
        self, 
        batch: CleopatraEnsembleInput, 
        step_type: str = "train"
    ) -> CleopatraOut:

        losses: list[Tensor] = []

        img, label, attention_mask, _ = batch
        ensemble_logits, learner_logits, additional_logs = self.forward(
            batch_lst=img, 
            attention_mask=attention_mask
        )
        models_logits: Tensor = torch.cat([ensemble_logits.unsqueeze(dim=1), learner_logits], dim=1)

        for lr_logits_idx in range(models_logits.shape[1]): 

            if self.hparams.use_weighted_loss: 
                weights = self.loss_weights if step_type == "train" else torch.ones_like(self.loss_weights)
                loss: Tensor = F.cross_entropy(
                        input=models_logits[:, lr_logits_idx], 
                        target=label, 
                        weight=weights
                    ) 

            else:
                loss: Tensor = F.cross_entropy(
                    input=models_logits[:, lr_logits_idx], 
                    target=label
                ) 

            losses.append(loss)

        if additional_logs is not None:
            self.log_dict(
                additional_logs, 
                on_step=True, 
                on_epoch=True
            )
                

        preds: Tensor = torch.argmax(
            models_logits, 
            dim=2
        )

        # Pos 0 reserved to the ensemble method
        return CleopatraOut(
            loss=torch.stack(losses), 
            logits=models_logits, 
            prediction=preds, 
            label=label
        )

    def training_step(self, batch: CleopatraEnsembleInput, batch_idx: int) -> Tensor:
        loss, _, _, _ = self.base_step(batch)
        self.log(
            name=f"{self.hparams.handler_model_name}_train_loss", 
            value=loss[0], 
            prog_bar=True, 
            on_step=True, 
            on_epoch=True
        )

        for idx, ls in enumerate(loss[1:]):
            self.log(
                name=f"{self.learners_name[idx]}_train_loss", 
                value=ls, 
                prog_bar=False, 
                on_step=False, 
                on_epoch=True
            )
        
        if self.current_epoch < self.hparams.min_epoch_handler_model:
            return loss[0]
        else:
            learner_loss_avg: Tensor = loss[1:].mean()

            return loss[0] + learner_loss_avg * self.hparams.learner_loss_regulizer


    def validation_step(self, batch: CleopatraEnsembleInput, batch_idx: int) -> None:
        loss, logits, preds, labels = self.base_step(
            batch=batch, 
            step_type="val"
        )

        self.log(
            name=f"{self.hparams.handler_model_name}_val_loss", 
            value=loss[0], 
            prog_bar=True, 
            on_step=False, 
            on_epoch=True
        )

        for idx in range(len(self.val_metrics)):
            # update metrics
            self.val_metrics[idx]["acc"].update(preds[:, idx], labels)
            self.val_metrics[idx]["f1"].update(preds[:, idx], labels)
            self.val_metrics[idx]["auc"].update(logits[:, idx], labels)

            if idx > 0:
                self.log(
                    f"{self.learners_name[idx - 1]}_val_loss", 
                    loss[idx], 
                    prog_bar=False, 
                    on_step=True, 
                    on_epoch=True
                )

    def on_validation_epoch_end(self) -> None:

        ensemble_vals: dict[str, Tensor] = self.val_metrics[0].compute()
        self.log_dict(
            {
                f"{self.hparams.handler_model_name}_val_acc": ensemble_vals["acc"],
                f"{self.hparams.handler_model_name}_val_f1":  ensemble_vals["f1"],
                f"{self.hparams.handler_model_name}_val_auc": ensemble_vals["auc"],
            },
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

        for idx in range(1, len(self.val_metrics)):
            lr_vals: dict[str, Tensor] = self.val_metrics[idx].compute()
            model_idx: int = idx - 1

            self.log_dict(
                {
                    f"{self.learners_name[model_idx]}_val_acc": lr_vals["acc"],
                    f"{self.learners_name[model_idx]}_val_f1":  lr_vals["f1"],
                    f"{self.learners_name[model_idx]}_val_auc": lr_vals["auc"],
                },
                prog_bar=False,
                on_epoch=True,
                on_step=False,
            )

        # reset all metrics
        for mc in self.val_metrics:
            mc.reset()

        # unfreeze if needed the learners block by block
        for lr_idx, learner in enumerate(self.learners):
            learner.unfreezing_handler(
                val_loss_name=f"{self.learners_name[lr_idx]}_val_loss", 
                log_blocks=False, 
                min_epoch=self.hparams.min_epoch_handler_model
            )

    def on_train_end(self) -> None:
        for learner in self.learners:
            learner.on_train_end()

        

