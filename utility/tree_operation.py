import torch
from utility.patch_shap_bpt import BPT
from torch import Tensor, tensor, zeros


def get_partition_lca_from_level(tree: BPT, level: int) -> list[list[float]]:
    for bpt_level in BPT.levels:
        if bpt_level.level_id == level:
            return [list(elem.coalition_member) for elem in bpt_level.nodes]
        
    raise Exception(
        f"The tree does not have the required level: {level}, "
        f"the actual height of the tree is {len(tree.levels)}"
    )
        
def get_parition_lca_from_percentage(tree: BPT, percentage: float = 0.3, min_margin: float = 0.1) -> list[list[float]]:
    patch_num: int = tree.total_leaves
    for bpt_level in tree.levels:
        level_cardinality: float = len(bpt_level.nodes)
        level_percentage: float = level_cardinality / patch_num

        if percentage - min_margin <= level_percentage <= percentage + min_margin:
            return [list(elem.coalition_member) for elem in bpt_level.nodes]
        
    raise Exception(
        f"No level found with this percentage: {percentage}, "
        f"try with a bigger min_margin w.r.t the actual: {min_margin}"
    )


def get_adjacency_pair_from_coalitions(data: list[list[float]], seq_size: int = 196) -> Tensor:
    adjacency: Tensor = zeros(seq_size + 1, seq_size + 1, dtype=torch.float)
    for coalition in data:
        coalition_t: Tensor = tensor(coalition)
        coalition_t += 1 # shift patches idx to make space for <CLS>

        adjacency[coalition_t.unsqueeze(dim=1), coalition_t.unsqueeze(dim=0)] = 1

    # Connect <CLS> token to all the patches
    adjacency[0, :] = 1.
    adjacency[:, 0] = 1

    return adjacency



def get_adjacency_from_BPT(
    tree: BPT,
    percentage: float = 0.3,
    margin: float = 0.1
) -> Tensor:
    
    data: list[list[float]] = get_parition_lca_from_percentage(
        tree=tree,
        percentage=percentage,
        min_margin=margin
    )

    return get_adjacency_pair_from_coalitions(
        data=data, 
        seq_size=tree.total_leaves
    )