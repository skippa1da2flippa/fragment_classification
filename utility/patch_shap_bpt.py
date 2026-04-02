from dataclasses import dataclass
from typing import Literal, NamedTuple
from torch import Tensor
from math import sqrt

@dataclass
class PatchWrapper: # it's a node in bpt tree
    colation_type: Literal["patch", "coalition"]
    coalition_id: int
    max_R: float
    min_R: float
    max_G: float
    min_G: float        
    max_B: float
    min_B: float
    color_range: float = -1
    perimeter: float 
    area: float
    lv: int 
    coalition_member: set[int]
    adjcent_coalition: set[int]

@dataclass
class DistanceWrapper:
    fst_coal: PatchWrapper
    sdn_coal: PatchWrapper
    distance: float

@dataclass
class BPT_level:
    level_id: int
    nodes: list[PatchWrapper]
    min_node_id: int
    max_node_id: int

@dataclass
class BPT:
    levels: list[BPT_level]


def color_range_f(r_channel: float, g_channel: float, b_channel: float) -> float:
    return r_channel ** 2 + g_channel ** 2 + b_channel ** 2

def get_patch_area(patch_size: int = 16) -> float:
    patch_size ** 2

def get_patch_perimeter(patch_size: int = 16) -> float:
    patch_size * 4 

def get_patch_distance(
    r_channel: float, 
    g_channel: float, 
    b_channel: float, 
    patch_size: int = 16    
) -> float:
    
    color_range: float = color_range_f(
        r_channel=r_channel, 
        g_channel=g_channel, 
        b_channel=b_channel
    )

    area: float = get_patch_area(
        patch_size=patch_size
    )
    perimeter: float = get_patch_perimeter(
        patch_size=patch_size
    )

    return color_range * area * sqrt(perimeter)

def get_common_perimeter(
    fst_coal: PatchWrapper, 
    sdn_coal: PatchWrapper,
    window_size: int = 16, 
    num_patches: int = 196
) -> float:
    
    fst_presence_lst: set[int] = fst_coal.adjcent_coalition.copy()
    sdn_presence_lst: set[int] = sdn_coal.adjcent_coalition.copy()

    fst_presence_lst.add(fst_coal.coalition_id)
    sdn_presence_lst.add(sdn_coal.coalition_id)

    coalition_common: set[int] = fst_presence_lst.intersection(
        sdn_presence_lst
    )
    patches_common: set[int] = coalition_common.intersection(
        list(range(num_patches))
    )
    patches_common = patches_common.intersection(
        coalition_common
    )
    
    common_perimeter: float = len(
        patches_common
    ) * window_size

    return common_perimeter


def get_coalition_distance(
    fst_coal: PatchWrapper, 
    sdn_coal: PatchWrapper
) -> float:
    
    red: float = max(
        fst_coal.max_R, sdn_coal.max_R
    ) - min(
        fst_coal.min_R, sdn_coal.min_R
    )

    green: float = max(
        fst_coal.max_G, sdn_coal.max_G
    ) - min(
        fst_coal.min_G, sdn_coal.min_G
    )

    blue: float = max(
        fst_coal.max_B, sdn_coal.max_B
    ) - min(
        fst_coal.min_B, sdn_coal.min_B
    )

    color_range: float = color_range_f(
        r_channel=red, 
        g_channel=green,
        b_channel=blue
    )

    area: float = fst_coal.area + sdn_coal.area

    perimeter: float = fst_coal.perimeter + sdn_coal.perimeter
    common_perimeter: float = get_common_perimeter(
        fst_coal=fst_coal,
        sdn_coal=sdn_coal
    )
    perimeter -= common_perimeter

    return color_range * area * sqrt(perimeter)


def from_one_to_double_coord(idx: int, max_row: int = 14) -> tuple[int, int]:
    if max_row**2 <= idx:
        raise ValueError(
            f"the received index: {idx}" 
            "does not lie within the specified boundaries: "
            f"{max_row**2}"
        )

    row_idx: int = idx // max_row
    col_idx: int = idx - (row_idx * max_row)

    return row_idx, col_idx

def from_double_to_one_coord(coord: tuple[int, int], max_row: int = 14) -> int:
    x_coord, y_coord = coord

    return (x_coord * max_row) + y_coord


def remove_negative_coord(lst: list[tuple[int, int]]) -> list[tuple[int, int]]:
    final_coord: list[tuple[int, int]] = []

    for idx in len(range(lst)):
        x_coord, y_coord = lst[idx]

        if x_coord >= 0 and y_coord >= 0:
            final_coord.append(lst[idx])

    return final_coord


def get_adjcent_patch_ids(actual_id: int, n_patch: int 14) -> list[int]:
    res: list[tuple[int, int]] = []
    x_coord, y_coord = from_one_to_double_coord(
        idx=actual_id, 
        max_row=n_patch
    )

    for idx in range(x_coord - 1, x_coord + 2, 2):
        for idj in range(y_coord - 1, y_coord + 2, 2):
            res.append((
                idx, idj
            ))

    filtered: list[tuple[int, int]] = remove_negative_coord(
        lst=res
    )

    return [from_double_to_one_coord(coord) for coord in filtered]

def get_candidate_from_adj(
    candidates: list[PatchWrapper], 
    adjacent_ids: list[int]
) -> list[PatchWrapper]:
    chosen: list[PatchWrapper] = []

    for adj_id in adjacent_ids:
        for cand in candidates:
            if adj_id == cand.coalition_id:
                chosen.append(
                    cand
                )

    return chosen

def find_best_pair(
    source: PatchWrapper, 
    candidates: list[PatchWrapper]
) -> list[tuple[DistanceWrapper, float]]:
    res_helder: list = []
    
    for candidate in candidates:
        distance: float = get_coalition_distance(
            fst_coal=source, 
            sdn_coal=candidate
        )
        
        res_helder.append(
            (candidate, distance)
        )

    res_helder = sorted(res_helder, key=lambda x: x[1])
        
    return res_helder


def handle_duplication(
    source_collector: dict[int, list[PatchWrapper]],
    coal_collector: dict[int, list[PatchWrapper]] 
) -> dict[int, PatchWrapper]:
    
    source_final: dict[int, PatchWrapper] = []
    coal_final: dict[int, PatchWrapper] = []

    for key in source_collector:
        sources: list[PatchWrapper] = source_collector[key]
        source_final[key] = sources[0]

        coal: list[PatchWrapper] = coal_collector[key]
        coal_final[key] = coal[0]

    

    sorted_source = sorted(source_final.values(), key=lambda x: x.coalition_id)

    pass

def get_chosen_pair(bpt_level: BPT_level) -> dict[int, PatchWrapper]:
    coalitions: list[PatchWrapper] = bpt_level.nodes
    source_collector: dict[int, list[PatchWrapper]] = []
    coal_collector: dict[int, list[PatchWrapper]] = []
    source_final: dict[int, PatchWrapper] = []

    for coalition in coalitions:
        adj: list[int] = coalition.adjcent_coalition

        candidates: list[PatchWrapper] = get_candidate_from_adj(
            candidates=coalitions,
            adjacent_ids=adj
        )

        sorted_candidates: list[tuple[PatchWrapper, float]] = find_best_pair(
            source=coalition, 
            candidates=candidates
        )

        coal_collector[coalition.coalition_id] = sorted_candidates

        for sort_cand, distance in sorted_candidates:
            if sort_cand.coalition_id in source_collector:
                source_collector[sort_cand.coalition_id].append(
                    coalition.coalition_id, 
                    distance
                )
            else:
                source_collector[sort_cand.coalition_id] = [
                    (coalition.coalition_id, distance)
                ]

    source_final = handle_duplication(
        source_final=source_final,
        source_collector=source_collector
    )

    return source_final





def merge(fst_coalition: PatchWrapper, sdn_coalition: PatchWrapper) -> PatchWrapper:
    pass


def initialize_partitions(patches: list[Tensor]) -> list[PatchWrapper]:
    res: list[PatchWrapper] = []

    for patch_id, patch in enumerate(patches):
        flattened_patch: Tensor = patch.flatten(start_dim=1)
        max_flat: Tensor = flattened_patch.max(dim=1)
        min_flat: Tensor = flattened_patch.min(dim=1)

        res.append(
            PatchWrapper(
                colation_type="patch", 
                colation_id=patch_id, 
                max_R=max_flat[0], 
                min_R=min_flat[0], 
                max_G=max_flat[1], 
                min_G=min_flat[1], 
                max_B=max_flat[2], 
                min_B=min_flat[2], 
                area=flattened_patch.shape[1],
                perimeter=patch.shape[1] * 4, 
                lv=len(patches), 
                coalition_member=set([patch_id]),
                adjcent_coalition=get_adjcent_patch_ids(patch_id)
            )
        )

    return res



def compute_color_range(fst_patch: Tensor, sdn_patch: Tensor) -> float:
    """
    Computes the color range between two patches.
    
    This function calculates the mean absolute difference in pixel values between two patches, 
    which can be used as a measure of color range or variation between the two patches.
    
    Args:
        fst_patch (Tensor): The first patch tensor of shape (C, H, W).
        sdn_patch (Tensor): The second patch tensor of shape (C, H, W).
    
    Returns:
        float: The computed color range between the two patches.
    """
    # Ensure the input tensors have the same shape
    if fst_patch.shape != sdn_patch.shape:
        raise ValueError("Input patches must have the same shape.")
    
    red_fst: Tensor = fst_patch[:, 0]
    red_sdn: Tensor = sdn_patch[:, 0]
    green_fst: Tensor = fst_patch[:, 1]
    green_sdn: Tensor = sdn_patch[:, 1]
    blue_fst: Tensor = fst_patch[:, 2]
    blue_sdn: Tensor = sdn_patch[:, 2]
    
    flt_fst_patch: Tensor = fst_patch.flatten(start_dim=2)
    flt_sdn_patch: Tensor = sdn_patch.flatten(start_dim=2)
    
    max_rgb: Tensor = flt_fst_patch.max(dim=2) - flt_sdn_patch.max(dim=2)
    min_rgb: Tensor = flt_fst_patch.min(dim=2) - flt_sdn_patch.min(dim=2)
  



# 1) Initialize the leaves with the patches wrapped in the object class `PatchWrapper` 
#    (takes as input a list of patches and return a list of `PatchWrapper`)
# 2) for each coaltion find the adjacent colations 
#    (takes as input a `PatchWrapper` and populate the field `PatchWrapper`.adjacent_coalition)
# 3) given the result at step 2) minimize the dist fucntion and create the new merged
#    object `PatchWrapper`
# 4) go on like this until we reach the root


# The result should be:
# for each level of the tree we have a list of `PatchWrapper` object each representing
# a given node of the tree at the desired level
# so in the end we will have a `dict[int, list[PatchWrapper]]` which will represent a single fragment tree