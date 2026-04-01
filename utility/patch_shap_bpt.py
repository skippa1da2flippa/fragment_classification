from typing import Literal, NamedTuple
from torch import Tensor
from math import sqrt

class PatchWrapper(NamedTuple): # it's a node in bpt tree
    colation_type: Literal["patch", "coalition"]
    coalition_id: int
    max_R: float
    min_R: float
    max_G: float
    min_G: float        
    max_B: float
    min_B: float
    color_range: float
    perimeter: float 
    area: float
    lv: int 
    coalition_member: set[int] # id of the patch
    adjcent_coalition: set[int]

class BPT_level: # return this from function leaves are 0
    level_id: int
    nodes: list[PatchWrapper]

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

    for idx in range(len(lst)):
        x_coord, y_coord = lst[idx]

        if x_coord >= 0 and y_coord >= 0:
            final_coord.append(lst[idx])

    return final_coord


def get_adjcent_patch_ids(actual_id: int, n_patch: int = 14) -> list[int]:
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

def get_chosen_pair(coalitions: list[PatchWrapper]):
    for coalition in coalitions:
        adj: list[int] = coalition.adjcent_coalition

def merge(fst_coalition: PatchWrapper, sdn_coalition: PatchWrapper) -> PatchWrapper:
    pass

def get_adjacent_coalition(coalitions: list[PatchWrapper]) -> list[int]:
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
                set=set([])
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

def leaves_wrapper(patches: list) -> list[PatchWrapper]:
    wrappers = []
    for patch_id, patch in enumerate(patches):
        wrapper = PatchWrapper(
            colation_type="patch",
            coalition_id= patch_id,
            max_R= patch[0].max().item(),
            min_R= patch[0].min().item(),
            max_G= patch[1].max().item(),
            min_G= patch[1].min().item(),        
            max_B= patch[2].max().item(),
            min_B= patch[2].min().item(),
            color_range = -1,
            perimeter=patch.shape[1]*patch.shape[2], 
            area= patch.shape[1]*4,
            lv= 1,
            coalition_member= {patch_id}, # id of the patch
            adjcent_coalition= set(get_adjcent_patch_ids(actual_id = patch_id))
        )
        wrappers.append(wrapper)
    return wrappers

# 2) for each coaltion find the adjacent colations 
#    (takes as input a `PatchWrapper` and populate the field `PatchWrapper`.adjacent_coalition)
# 3) given the result at step 2) minimize the dist fucntion and create the new merged
#    object `PatchWrapper`
# 4) go on like this until we reach the root


# The result should be:
# for each level of the tree we have a list of `PatchWrapper` object each representing
# a given node of the tree at the desired level
# so in the end we will have a `dict[int, list[PatchWrapper]]` which will represent a single fragment tree