from timm.models.vision_transformer import VisionTransformer
from typing import Optional, Type, Union, Tuple, Literal, Callable
from timm.layers import PatchEmbed
from timm.models.vision_transformer import Block
from timm.layers import (LayerType, Mlp, Attention)
import torch
from torch import Tensor
import torch.nn as nn

from models_handler.layer.super_block import SuperBlock


class GraphVit(VisionTransformer):
    def __init__(
        self,
        gnn_type: str,
        gnn_num_layer: int = 1,
        pretrained: bool = True,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: Literal['', 'avg', 'avgmax', 'max', 'token', 'map'] = 'token',
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        scale_attn_norm: bool = False,
        scale_mlp_norm: bool = False,
        proj_bias: bool = True,
        init_values: Optional[float] = None,
        class_token: bool = True,
        pos_embed: str = 'learn',
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        final_norm: bool = True,
        fc_norm: Optional[bool] = None,
        pool_include_prefix: bool = False,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        drop_rate: float = 0.,
        pos_drop_rate: float = 0.,
        patch_drop_rate: float = 0.,
        proj_drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        weight_init: Literal['skip', 'jax', 'jax_nlhb', 'moco', ''] = '',
        fix_init: bool = False,
        embed_layer: Callable = PatchEmbed,
        embed_norm_layer: Optional[LayerType] = None,
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        block_fn: Type[nn.Module] = Block,
        mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:

        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool=global_pool,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            scale_attn_norm=scale_attn_norm,
            scale_mlp_norm=scale_mlp_norm,
            proj_bias=proj_bias,
            init_values=init_values,
            class_token=class_token,
            pos_embed=pos_embed,
            no_embed_class=no_embed_class,
            reg_tokens=reg_tokens,
            pre_norm=pre_norm,
            final_norm=final_norm,
            fc_norm=fc_norm,
            pool_include_prefix=pool_include_prefix,
            dynamic_img_size=dynamic_img_size,
            dynamic_img_pad=dynamic_img_pad,
            drop_rate=drop_rate,
            pos_drop_rate=pos_drop_rate,
            patch_drop_rate=patch_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            weight_init=weight_init,
            fix_init=fix_init,
            embed_layer=embed_layer,
            embed_norm_layer=embed_norm_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            block_fn=block_fn,
            mlp_layer=mlp_layer  
        )

        self.gnn_type: str = gnn_type
        self.gnn_num_layer: int = gnn_num_layer
        self.pretrained: bool = pretrained

        self.num_heads = num_heads 
        self.mlp_ratio = mlp_ratio
        self.qkv_bias= qkv_bias 
        self.qk_norm = qk_norm 
        self.scale_attn_norm = scale_attn_norm 
        self.scale_mlp_norm = scale_mlp_norm 
        self.proj_bias = proj_bias 
        self.proj_drop = proj_drop_rate 
        self.attn_drop = attn_drop_rate 
        self.init_values = init_values 
        self.drop_path = drop_path_rate 
        self.act_layer = act_layer 
        self.norm_layer = norm_layer 
        self.mlp_layer = mlp_layer

        if not self.pretrained:
            self._deconstruct_model()

    @staticmethod
    def build_from_vision_transformer(
        vit_model: VisionTransformer,
        gnn_type: str,
        gnn_num_layer: int = 1,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: Literal['', 'avg', 'avgmax', 'max', 'token', 'map'] = 'token',
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        scale_attn_norm: bool = False,
        scale_mlp_norm: bool = False,
        proj_bias: bool = True,
        init_values: Optional[float] = None,
        class_token: bool = True,
        pos_embed: str = 'learn',
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        final_norm: bool = True,
        fc_norm: Optional[bool] = None,
        pool_include_prefix: bool = False,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        drop_rate: float = 0.,
        pos_drop_rate: float = 0.,
        patch_drop_rate: float = 0.,
        proj_drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        weight_init: Literal['skip', 'jax', 'jax_nlhb', 'moco', ''] = '',
        fix_init: bool = False,
        embed_layer: Callable = PatchEmbed,
        embed_norm_layer: Optional[LayerType] = None,
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        block_fn: Type[nn.Module] = Block,
        mlp_layer: Type[nn.Module] = Mlp
    ) -> 'GraphVit':
        
        graph_vit: 'GraphVit' = GraphVit(
            gnn_type=gnn_type,
            gnn_num_layer=gnn_num_layer,
            pretrained=True,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool=global_pool,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            scale_attn_norm=scale_attn_norm,
            scale_mlp_norm=scale_mlp_norm,
            proj_bias=proj_bias,
            init_values=init_values,
            class_token=class_token,
            pos_embed=pos_embed,
            no_embed_class=no_embed_class,
            reg_tokens=reg_tokens,
            pre_norm=pre_norm,
            final_norm=final_norm,
            fc_norm=fc_norm,
            pool_include_prefix=pool_include_prefix,
            dynamic_img_size=dynamic_img_size,
            dynamic_img_pad=dynamic_img_pad,
            drop_rate=drop_rate,
            pos_drop_rate=pos_drop_rate,
            patch_drop_rate=patch_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            weight_init=weight_init,
            fix_init=fix_init,
            embed_layer=embed_layer,
            embed_norm_layer=embed_norm_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            block_fn=block_fn,
            mlp_layer=mlp_layer  
        )

        graph_vit.patch_embed = vit_model.patch_embed
        graph_vit.pos_drop = vit_model.pos_drop
        graph_vit.patch_drop = vit_model.patch_drop
        graph_vit.norm_pre = vit_model.norm_pre
        graph_vit.blocks = vit_model.blocks
        graph_vit.norm = vit_model.norm
        graph_vit.fc_norm = vit_model.fc_norm
        graph_vit.head_drop = vit_model.head_drop
        graph_vit.head = vit_model.head

        graph_vit._deconstruct_model()

        return graph_vit

    def _deconstruct_model(self):
        for block_idx in range(len(self.blocks)):
            new_block: SuperBlock = SuperBlock(
                gnn_type=self.gnn_type,
                gnn_num_layer=self.gnn_num_layer,
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_norm=self.qk_norm,
                scale_attn_norm=self.scale_attn_norm,
                scale_mlp_norm=self.scale_mlp_norm,
                proj_bias=self.proj_bias,
                proj_drop=self.proj_drop,
                attn_drop=self.attn_drop,
                init_values=self.init_values,
                drop_path=self.drop_path
            )

            if self.pretrained:
                new_block.norm1 = self.blocks[block_idx].norm1
                new_block.attn = self.blocks[block_idx].attn
                new_block.ls1 = self.blocks[block_idx].ls1
                new_block.drop_path1 = self.blocks[block_idx].drop_path1
                new_block.norm2 = self.blocks[block_idx].norm2
                new_block.mlp = self.blocks[block_idx].mlp
                new_block.ls2 = self.blocks[block_idx].ls2
                new_block.drop_path2 = self.blocks[block_idx].drop_path2

            self.blocks[block_idx] = new_block

    def forward_features(
            self,
            x: Tensor,
            bpt_partitions: Tensor | None = None,
            attn_mask: Optional[Tensor] = None,
            is_causal: bool = False,
    ) -> Tensor:
        """Forward pass through feature layers (embeddings, transformer blocks, post-transformer norm)."""
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        vit_input: dict = {
            "x": x
        }

        if attn_mask is not None or is_causal:
            vit_input["attn_mask"] = attn_mask
            vit_input["is_causal"] = is_causal

        if bpt_partitions is not None:
            vit_input["bpt_partitions"] = bpt_partitions

        # If mask/causal provided, we need to apply blocks one by one
        if len(vit_input) > 1:
            for blk in self.blocks:
                vit_input["x"] = blk(**vit_input)

        # elif self.grad_checkpointing and not torch.jit.is_scripting():
        #     x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)

        x = self.norm(x)
        return x


    def forward(
        self,
        x: Tensor,
        bpt_partitions: Tensor | None = None,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False
    ) -> Tensor:
        x = self.forward_features(x, bpt_partitions=bpt_partitions, attn_mask=attn_mask, is_causal=is_causal)
        x = self.forward_head(x)
        return x