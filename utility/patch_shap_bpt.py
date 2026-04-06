import json
from dataclasses import dataclass
from typing import Literal
from torch import Tensor
from math import sqrt
from tqdm import tqdm
from utility.utility import get_splitted_image, load_from_image_to_tensor, load_image

@dataclass
class PatchWrapper: # it's a node in bpt tree
    colation_type: Literal["patch", "coalition"]
    coalition_id: int
    coalition_member: set[int]
    adjacent_coalition: set[int]
    max_R: float
    min_R: float
    max_G: float
    min_G: float        
    max_B: float
    min_B: float
    perimeter: float 
    area: float
    color_range: float = -1
    lv: int | None = None
    kids: tuple[int, int] | None = None 

    def __eq__(self, value: "PatchWrapper") -> bool:
        return self.coalition_id == value.coalition_id
    
    def __len__(self) -> int:
        return len(self.coalition_member)
    
    def copy(self) -> "PatchWrapper":
        return PatchWrapper(
            colation_type=self.colation_type,
            coalition_id=self.coalition_id,
            coalition_member=self.coalition_member.copy(),
            adjacent_coalition=self.adjacent_coalition.copy(),
            max_R=self.max_R,
            min_R=self.min_R,
            max_G=self.max_G,
            min_G=self.min_G,
            max_B=self.max_B,
            min_B=self.min_B,
            perimeter=self.perimeter,
            area=self.area,
            color_range=self.color_range,
            lv=self.lv,
            kids=tuple(self.kids) if self.kids is not None else None
        )

    def to_dict(self) -> dict:
        return {
            "colation_type": self.colation_type,
            "coalition_id": self.coalition_id,
            "coalition_member": self.coalition_member,
            "adjacent_coalition": self.adjacent_coalition,
            "max_R": self.max_R,
            "min_R": self.min_R,
            "max_G": self.max_G,
            "min_G": self.min_G,
            "max_B": self.max_B,
            "min_B": self.min_B,
            "perimeter": self.perimeter,
            "area": self.area,
            "color_range": self.color_range,
            "lv": self.lv,
            "kids": list(self.kids) if self.kids is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PatchWrapper":
        return cls(
            colation_type=data["colation_type"],
            coalition_id=data["coalition_id"],
            coalition_member=set(data["coalition_member"]),
            adjacent_coalition=set(data["adjacent_coalition"]),
            max_R=data["max_R"],
            min_R=data["min_R"],
            max_G=data["max_G"],
            min_G=data["min_G"],
            max_B=data["max_B"],
            min_B=data["min_B"],
            perimeter=data["perimeter"],
            area=data["area"],
            color_range=data.get("color_range", -1),
            lv=data.get("lv"),
            kids=tuple(data["kids"]) if data.get("kids") is not None else None,
        )

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

    def to_dict(self) -> dict:
        return {
            "level_id": self.level_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "min_node_id": self.min_node_id,
            "max_node_id": self.max_node_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BPT_level":
        return cls(
            level_id=data["level_id"],
            nodes=[PatchWrapper.from_dict(node) for node in data["nodes"]],
            min_node_id=data["min_node_id"],
            max_node_id=data["max_node_id"],
        )

@dataclass
class BPT:
    levels: list[BPT_level]
    total_nodes: int 
    total_leaves: int
    height: int

    def to_dict(self) -> dict:
        return {
            "levels": [level.to_dict() for level in self.levels],
            "total_nodes": self.total_nodes,
            "total_leaves": self.total_leaves,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BPT":
        return cls(
            levels=[BPT_level.from_dict(level) for level in data["levels"]],
            total_nodes=data["total_nodes"],
            total_leaves=data["total_leaves"],
            height=data["height"],
        )


def save_bpt_to_json(bpt: BPT, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(bpt.to_dict(), fp, indent=2)


def load_bpt_from_json(path: str) -> BPT:
    with open(path, "r", encoding="utf-8") as fp:
        return BPT.from_dict(json.load(fp))


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
    
    fst_presence_lst: set[int] = fst_coal.adjacent_coalition.copy()
    sdn_presence_lst: set[int] = sdn_coal.adjacent_coalition.copy()

    fst_presence_lst.add(fst_coal.coalition_id)

    coalition_common: set[int] = fst_coal.coalition_member.intersection(
        sdn_coal.adjacent_coalition
    )
    patches_common: set[int] = coalition_common.intersection(
        list(range(num_patches))
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

    for idx in range(len(lst)):
        x_coord, y_coord = lst[idx]

        if x_coord >= 0 and y_coord >= 0:
            final_coord.append(lst[idx])

    return final_coord


def get_adjcent_patch_ids(actual_id: int, n_patch: int = 14) -> set[int]:
    res: list[tuple[int, int]] = []
    x_coord, y_coord = from_one_to_double_coord(
        idx=actual_id, 
        max_row=n_patch
    )

    for idx in range(x_coord - 1, x_coord + 2):
        for idj in range(y_coord - 1, y_coord + 2):
            if idx != x_coord or idj != y_coord:
                res.append((
                    idx, idj
                ))

    filtered: list[tuple[int, int]] = remove_negative_coord(
        lst=res
    )

    return set([from_double_to_one_coord(coord) for coord in filtered])

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
) -> list[tuple[PatchWrapper, float]]:
    res_helder: list[tuple[PatchWrapper, float]] = []
    
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


def get_merges(
    source_collector: dict[int, list[tuple[PatchWrapper, float]]],
    coal_collector: dict[int, list[tuple[PatchWrapper, float]]] 
) -> tuple[dict[int, tuple[PatchWrapper, PatchWrapper, float]], set[int]]:
    
    final_out: dict[int, tuple[PatchWrapper, PatchWrapper, float]] = {}
    source_final: dict[int, tuple[PatchWrapper, float]] = {}
    coal_final: dict[int, tuple[PatchWrapper, float]] = {}
    available_source: set[int] = set()

    max_coalition_id: int = max(coal_collector.keys())
    actual_coalition_id: int = max_coalition_id + 1

    for key in source_collector:
        sources: list[tuple[PatchWrapper, float]] = source_collector[key]
        source_final[key] = sources[0]

        coal: list[tuple[PatchWrapper, float]] = coal_collector[key]
        coal_final[key] = coal[0]

        available_source.add(key)

    for key in coal_final:
        candidate_coal: PatchWrapper = coal_final[key][0]
        candidate_source: PatchWrapper = source_final[candidate_coal.coalition_id][0]

        if key not in available_source:
            continue

        if candidate_source.coalition_id == key:
            available_source.remove(candidate_coal.coalition_id)
            available_source.remove(candidate_source.coalition_id)

            final_out[actual_coalition_id] = (candidate_source, candidate_coal, coal_final[key][1])
            actual_coalition_id += 1
        
        else:
            for fall_back_cand, fall_back_distance in coal_collector[key][1:]:
                fall_back_coal: PatchWrapper = fall_back_cand
                source_fall_back: PatchWrapper = source_final[fall_back_coal.coalition_id][0]

                if fall_back_coal.coalition_id in available_source:
                    if source_fall_back.coalition_id in available_source:
                        if source_fall_back.coalition_id == key:
                            available_source.remove(fall_back_coal.coalition_id)
                            available_source.remove(source_fall_back.coalition_id)

                            final_out[actual_coalition_id] = (source_fall_back, fall_back_coal, fall_back_distance)
                            actual_coalition_id += 1

                            break

    
    return final_out, available_source

def get_chosen_pair(bpt_level: BPT_level) -> tuple[dict[int, tuple[PatchWrapper, float]], list[PatchWrapper]]:
    coalitions: list[PatchWrapper] = bpt_level.nodes
    source_collector: dict[int, list[PatchWrapper]] = {}
    coal_collector: dict[int, list[PatchWrapper]] = {}

    for coalition in coalitions:
        adj: list[int] = coalition.adjacent_coalition

        # TODO ERRORE al level = 1 (definizione secondo livello) candidates è vuoto
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
                    (coalition, distance)
                )
            else:
                source_collector[sort_cand.coalition_id] = [
                    (coalition, distance)
                ]

    merges, available_source = get_merges(
        coal_collector=coal_collector,
        source_collector=source_collector
    )

    leftout_nodes: list[PatchWrapper] = []
    for coal_id in available_source:
        for coal in coalitions:
            if coal.coalition_id == coal_id:
                new_coal: PatchWrapper = coal.copy()
                new_coal.lv = coal.lv + 1 if coal.lv is not None else None
                leftout_nodes.append(
                    new_coal
                )

    return merges, leftout_nodes


def merge(
    fst_coalition: PatchWrapper, 
    sdn_coalition: PatchWrapper, 
    new_id: int, 
    color_distance: float
) -> PatchWrapper:

    coalition_member: set[int] = fst_coalition.coalition_member.union(
        sdn_coalition.coalition_member 
    )
    adjacent_coalition: set[int] = fst_coalition.adjacent_coalition.union(
        sdn_coalition.adjacent_coalition 
    )

    max_R: float = max(
        fst_coalition.max_R, sdn_coalition.max_R
    )
    min_R: float = min(
        fst_coalition.min_R, sdn_coalition.min_R
    )
    max_G: float = max(
        fst_coalition.max_G, sdn_coalition.max_G
    )
    min_G: float = min(
        fst_coalition.min_G, sdn_coalition.min_G
    )
    max_B: float = max(
        fst_coalition.max_B, sdn_coalition.max_B
    )
    min_B: float = min(
        fst_coalition.min_B, sdn_coalition.min_B
    )

    area: float = fst_coalition.area + sdn_coalition.area
    perimeter: float = fst_coalition.perimeter + sdn_coalition.perimeter
    common_perimeter: float = get_common_perimeter(
        fst_coal=fst_coalition,
        sdn_coal=sdn_coalition
    )
    perimeter -= 2*common_perimeter

    return PatchWrapper(
        colation_type="coalition",
        coalition_id=new_id,
        max_R=max_R,
        min_R=min_R,
        max_G=max_G,
        min_G=min_G,
        max_B=max_B,
        min_B=min_B,
        color_range=color_distance,
        perimeter=perimeter,
        area=area,
        coalition_member=coalition_member,
        adjacent_coalition=adjacent_coalition,
        kids=(fst_coalition.coalition_id, sdn_coalition.coalition_id), 
        lv=fst_coalition.lv + 1 if fst_coalition.lv is not None else None
    )

def get_new_level(bpt_level: BPT_level) -> BPT_level:
    merges, leftout_nodes = get_chosen_pair(
        bpt_level=bpt_level
    )

    new_nodes: list[PatchWrapper] = []

    for coal_id in merges:
        fst_coal, sdn_coal, color_distance = merges[coal_id]
        new_node: PatchWrapper = merge(
            fst_coalition=fst_coal,
            sdn_coalition=sdn_coal,
            new_id=coal_id,
            color_distance=color_distance
        )
        new_nodes.append(
            new_node
        )

    new_nodes.extend(leftout_nodes)

    return BPT_level(
        level_id=bpt_level.level_id + 1,
        nodes=new_nodes,
        min_node_id=min(new_nodes, key=lambda x: x.coalition_id).coalition_id,
        max_node_id=max(new_nodes, key=lambda x: x.coalition_id).coalition_id
    )

def initialize_partitions(patches: list[Tensor]) -> BPT_level:
    res: list[PatchWrapper] = []

    for patch_id, patch in enumerate(patches):
        flattened_patch: Tensor = patch.flatten(start_dim=1)
        max_flat: Tensor = flattened_patch.max(dim=1).values
        min_flat: Tensor = flattened_patch.min(dim=1).values

        res.append(
            PatchWrapper(
                colation_type="patch", 
                coalition_id=patch_id, 
                max_R=max_flat[0], 
                min_R=min_flat[0], 
                max_G=max_flat[1], 
                min_G=min_flat[1], 
                max_B=max_flat[2], 
                min_B=min_flat[2], 
                area=flattened_patch.shape[1],
                perimeter=patch.shape[1] * 4, 
                coalition_member=set([patch_id]),
                adjacent_coalition=get_adjcent_patch_ids(patch_id), 
                lv=0
            )
        )

    return BPT_level(
        level_id=0,
        nodes=res,
        min_node_id=0,
        max_node_id=len(patches) - 1
    )


def get_bpt_from_image(img_path: str) -> BPT:

    levels: list[BPT_level] = []
    total_nodes: int = 0

    # load image
    img: Tensor = load_from_image_to_tensor(
        img_path=img_path, 
        identity=True
    )[0]

    # split the image in 196 patches of 16x16
    patches: list[Tensor] = get_splitted_image(img)

    # call initialize_partitions to get the first level of the bpt
    level_zero: BPT_level = initialize_partitions(patches=patches)
    total_nodes += len(level_zero.nodes)
    levels.append(level_zero)

    out_cond: bool = False
    prev_level: BPT_level = level_zero
    initial_nodes: int = len(level_zero.nodes)
    
    with tqdm(total=initial_nodes - 1, desc="Building BPT levels") as progress:
        while not out_cond:
            new_level: BPT_level = get_new_level(
                bpt_level=prev_level
            )
            nodes_reduced: int = len(prev_level.nodes) - len(new_level.nodes)
            
            if len(new_level.nodes) == 1:
                out_cond = True
            
            levels = [new_level] + levels
            prev_level = new_level

            new_nodes_count: int = new_level.max_node_id - prev_level.min_node_id
            total_nodes += new_nodes_count
            progress.update(nodes_reduced)
            progress.set_description(f"Building BPT level-{len(levels) - 1} ({new_nodes_count} nodes)")


    return BPT(
        levels=levels, 
        total_nodes=total_nodes, 
        total_leaves=len(level_zero.nodes), 
        height=len(levels) - 1
    )


def single_pipeline_bpt(in_img_path: str, out_json_path: str) -> None:
    bpt: BPT = get_bpt_from_image(img_path=in_img_path)
    save_bpt_to_json(
        bpt=bpt, 
        out_json_path=out_json_path
    )
