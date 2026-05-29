import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch

from utility.utility import GNNType, GraphGenout, generate_connection

class GraphLocalAttention(nn.Module):
    def __init__(
        self, 
        gnn_type: str, 
        in_channels: int = 768,
        hidden_channels: int = 768,
        gnn_num_layer: int = 1, 
        cosine_temperature: float = 1.,
        cosine_threshold: float = 0.7, 
        pruned: bool = True
    ) -> None:
        super().__init__()

        self.gnn: nn.Module = GNNType[gnn_type].value(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=gnn_num_layer
        )
        self.cosine_threshold: float = cosine_threshold
        self.cosine_temperature: float = cosine_temperature
        self.pruned: bool = pruned

    def forward(
        self, 
        x: Tensor, 
        attn_mask: Tensor | None = None, 
        bpt_partitions: Tensor | None = None
    ) -> Tensor:
        patches: Tensor = x

        graph_genout: GraphGenout = generate_connection(
            patches_emb=patches,
            bpt_adjacency=bpt_partitions,
            threshold=self.cosine_threshold,
            valid_patch_mask=attn_mask.unsqueeze(1) if attn_mask is not None else None, 
            device=patches.device, 
            temperature=self.cosine_temperature
        )

        graph_batch: Batch = Batch.from_data_list(graph_genout.graph_batch)
        patches_embedding: Tensor = self.gnn(
            x=graph_batch.x, 
            edge_index=graph_batch.edge_index, 
            batch=graph_batch.batch
        )

        return patches_embedding.view(x.size(0), -1, x.size(2))