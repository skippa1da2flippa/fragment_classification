from utility.patch_shap_bpt import BPT


def get_partition_lca_from_level(tree: BPT, level: int) -> list[list[float]]:
    for bpt_level in BPT.levels:
        if bpt_level.level_id == level:
            return [list(elem.coalition_member) for elem in bpt_level.nodes]
        
    raise Exception(
        f"The tree does not have the required level: {level}, "
        f"the actual height of the tree is {len(tree.levels)}"
    )
        
def get_parition_lca_from_percentage(tree: BPT, percentage: float, min_margin: float = 0.05) -> list[list[float]]:
    patch_num: int = tree.total_leaves
    for bpt_level in BPT.levels:
        level_cardinality: float = len(bpt_level.nodes)
        level_percentage: float = level_cardinality / patch_num

        if percentage - min_margin <= level_percentage <= percentage + min_margin:
            return [list(elem.coalition_member) for elem in bpt_level.nodes]
        
    raise Exception(
        f"No level found with this percentage: {percentage}, "
        f"try with a bigger min_margin w.r.t the actual: {min_margin}"
    )
