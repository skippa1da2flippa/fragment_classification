import torch
import torch.nn as nn
from torch import Tensor
from typing import Literal, Type
from models_handler.base.base_ensemble import BaseEnsemble
from models_handler.base.base_learner import BaseLearner
from utility.utility import EnsembleForwardOut, GNNType, GraphGenout, generate_connection_discrete, get_basked_representation, multiple_generate_connection_discrete
from torch_geometric.data import Batch
import torch.nn.functional as F

class GraphEnsemble(BaseEnsemble):
    def __init__(
        self, 
        model_paths: list[tuple[str, str]], 
        model_types: list[Type[BaseLearner]] | Type[BaseLearner],
        model_dataset_info: list[int],
        gnn_type: str, 
        gnn_num_layer: int,
        learner_loss_regulizer: float = 0.2,
        decision_mode: Literal["least", "most", "all"] = "most",
        learners_name: list[str] = [],
        min_epoch_gnn: int = 5,
        central_node_mode: str = "zero",
        initial_emb_size: int = 768,
        final_head_size: int = 11,
        gnn_act_fun: str = "relu",
        gnn_dropout: float = 0.2,
        graph_load_param: float = 0.7,
        use_weighted_loss: bool = False,
        full_dataset: bool = True,
        lr: float = 0.01,
        weight_decay: float = 0.003,
        mask_on_learner: int = 2,
        temperature: float = 0.9, 
        edge_creation_mode: Literal["center", "upper"] = "center",
        cosine_threshold: float = 0.7,
        keep_temperature_stable: bool = False
    ) -> None:
        
        gnn: nn.Module = GNNType[gnn_type].value(
            in_channels=initial_emb_size,
            hidden_channels=initial_emb_size,
            num_layers=gnn_num_layer,
            out_channels=final_head_size, 
            act=gnn_act_fun, 
            dropout=gnn_dropout
        )
        
        super().__init__(
            model_paths=model_paths,
            model_types=model_types,
            handler_model=gnn,
            handler_model_name=GNNType[gnn_type].name,
            learners_name=learners_name,
            min_epoch_handler_model=min_epoch_gnn,
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
                "decision_mode": decision_mode, 
                "central_node_mode": central_node_mode, 
                "graph_load_param": graph_load_param, 
                "temperature": temperature,
                "gnn_type": gnn_type,
                "gnn_act_fun": gnn_act_fun, 
                "gnn_dropout": gnn_dropout,
                "model_dataset_info": model_dataset_info,
                "gnn_num_layer": gnn_num_layer, 
                "edge_creation_mode": edge_creation_mode,
                "cosine_threshold": cosine_threshold
            }
        )

        self.temperature: nn.Parameter = nn.Parameter(
            data=torch.tensor(
                temperature, 
                requires_grad=keep_temperature_stable
            )
        )


    def forward(
        self, 
        batch_lst: list[Tensor], 
        attention_mask: Tensor | None = None
    ) -> EnsembleForwardOut:
        
        valid_patch_mask: list[Tensor] = []
        step: int = attention_mask.shape[1] + len(self.learners)
        diagonal_att_mask: Tensor = attention_mask.diagonal(dim1=1, dim2=2) 

        learners_logits, learners_embedding = self.learners_forward(
            batch_lst=batch_lst,
            attention_mask=attention_mask, 
            return_tensor=(self.hparams.decision_mode != 'all'), 
            idx_lst=self.hparams.model_dataset_info
        )
        
        if self.hparams.decision_mode != 'all':
            chosen_lr_patches, other_patches, chosen_ids = get_basked_representation(
                ensemble_logits_t=learners_logits, 
                ensemble_patches_t=learners_embedding,
                choice=self.hparams.decision_mode
            )

            mask_map: Tensor = chosen_ids == self.hparams.mask_on_learner
            for idx, elem in enumerate(mask_map):
                if elem:
                    valid_patch_mask.append(
                        diagonal_att_mask[idx]
                    )
                else:
                    valid_patch_mask.append(
                        torch.ones_like(diagonal_att_mask[idx])
                    )

            graph_out: GraphGenout = generate_connection_discrete(
                patches_emb=chosen_lr_patches,
                other_global_nodes=other_patches[:, :, 0, :], 
                central_node_mode=self.hparams.central_node_mode,
                load_param=self.hparams.graph_load_param, 
                temperature=self.temperature,
                valid_patch_mask=torch.stack(valid_patch_mask), 
                device=self.device, 
                adapt_load_param=False,
                edge_creation_mode=self.hparams.edge_creation_mode,
                threshold=self.hparams.cosine_threshold
            )

        else:
            graph_out: GraphGenout = multiple_generate_connection_discrete(
                patches_emb=learners_embedding, 
                load_param=self.hparams.graph_load_param, 
                temperature=self.temperature,
                valid_patch_mask=diagonal_att_mask, 
                device=self.device, 
                mask_on_learner=self.hparams.mask_on_learner, 
                central_node_mode=self.hparams.central_node_mode, 
                adapt_load_param=False,
                edge_creation_mode=self.hparams.edge_creation_mode,
                threshold=self.hparams.cosine_threshold
            )

            step = (attention_mask.shape[1] * len(self.learners)) + 1

        additional_log: dict = {
            "graph_edges_cardinality_mean": graph_out.graph_edges_cardinality.mean(), 
            "graph_edges_cardinality_std": graph_out.graph_edges_cardinality.std(),
            "graph_density_mean": graph_out.graph_density.mean(),
            "graph_density_std": graph_out.graph_density.std()
        }

        graph_batch: Batch = Batch.from_data_list(graph_out.graph_batch).to(self.device)
        logits: Tensor = self.handler_model(
            x=graph_batch.x,
            edge_index=graph_batch.edge_index, 
            batch=graph_batch.batch
        )

        # TODO you are a criminal you should have adapted the 
        # `multiple_generate_connection_discrete` to receives
        # tensors, so you could have avoided this thing 
        if isinstance(learners_logits, list):
            learners_logits = [en.unsqueeze(dim=1) for en in learners_logits]
            learners_logits: Tensor = torch.cat(learners_logits, dim=1)

        # Return just the logits related to the central vector 
        # (positioned in the end of the sequence) of each graph
        return EnsembleForwardOut(
           ensemble_logits=logits[step - 1::step, :],
           learners_logits=learners_logits, 
           additional_log=additional_log
        )
    
    
    def on_fit_start(self) -> None:
        for learner in self.learners:
            learner.trainer = self.trainer
    