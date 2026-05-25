from collections import OrderedDict

from timm.models.vision_transformer import Block
from timm.layers import (LayerNorm, LayerType, Mlp, Attention)
from torch import Tensor
import torch.nn as nn
from typing import Optional, Type

from models_handler.layer.graph_local_attention import GraphLocalAttention

class SuperBlock(Block):
    def __init__(
        self,
        gnn_type: str,
        gnn_num_layer: int,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_attn_norm: bool = False,
        scale_mlp_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.,
        attn_drop: float = 0.,
        init_values: Optional[float] = None,
        drop_path: float = 0.,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = LayerNorm,
        mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            scale_attn_norm=scale_attn_norm,
            scale_mlp_norm=scale_mlp_norm,
            proj_bias=proj_bias,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer
        )

        self.norm0: nn.Module = norm_layer(dim)
        self.local_attention: nn.Module = GraphLocalAttention(
            gnn_type=gnn_type,
            in_channels=dim,
            hidden_channels=dim,
            gnn_num_layer=gnn_num_layer,
            cosine_threshold=0.7, 
            pruned=True
        )

        # reorder modules for printing
        new_order = OrderedDict()
        new_order["norm0"] = self.norm0
        new_order["local_attention"] = self.local_attention

        for name, module in self._modules.items():
            if name not in ["norm0", "local_attention"]:
                new_order[name] = module

        self._modules = new_order
    

    def forward(
        self, x: Tensor, 
        attn_mask: Tensor | None = None, 
        bpt_partitions: Tensor | None = None
    ) -> Tensor:
        x[:, 1:, :] = self.norm0(x[:, 1:, :])
        local_out: Tensor = self.local_attention(
            x=x,
            attn_mask=attn_mask,
            bpt_partitions=bpt_partitions
        )
        x[:, 1:, :] = x[:, 1:, :] + local_out

        return super().forward(x=x, attn_mask=attn_mask)