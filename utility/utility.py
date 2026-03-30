from enum import Enum
import os
from typing import Literal, NamedTuple
from torch import Tensor, bmm, ones
import torch
from torch_geometric.nn.models import GCN, GraphSAGE, GAT, GIN
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, Batch
from torchvision import transforms as T
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics as tm
from torchmetrics import MetricCollection


# ImageNet normalization
mean3 = [0.485, 0.456, 0.406]
mean4 = [0.485, 0.456, 0.406, 0] # Add 0 for the alpha channel
std3 = [0.229, 0.224, 0.225]
std4 = [0.229, 0.224, 0.225, 1] # Add 1 for the alpha channel

style_mapping = {
    "Byzantine": "middle",
    "Cubism": "modern",
    "Egyptian": "antique",
    "Etruscan": "antique",
    "Gothic": "middle",
    "Greek": "antique",
    "Impressionism": "modern",
    "Prehistory": "antique",
    "Renaissance": "middle",
    "Roman": "antique",
    "Surrealism": "modern"
}

# A function for reversing ImageNet normalization
denormalize = T.Normalize(mean=[-m/s for m, s in zip(mean3, std3)], std=[1/s for s in std3])

def train_transform(train_gnn: bool = True, alpha: bool = True): # TODO set to false train_gnn
    x_flip: float = .0 if train_gnn else .5
    y_flip: int = 0 if train_gnn else 20

    return T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(x_flip),
        T.RandomRotation(y_flip), 
        T.ToTensor(),
        # ImageNet normalization
        T.Normalize(mean=mean4, std=std4) if alpha else T.Normalize(mean=mean3, std=std3),
    ])
    
def eval_transform(alpha=True):
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        # ImageNet normalization
        T.Normalize(mean=mean4, std=std4) if alpha else T.Normalize(mean=mean3, std=std3),
    ])

def get_dataset_cardinality(dataset_path: str, full_count: bool = True) -> dict[str, int]:
    res: dict[str, int] = {}

    for split_name in os.listdir(dataset_path):
        if (
            (not full_count and split_name in {"valid", "test"}) 
            or 
            (full_count and split_name == "test")
        ):
            continue

        split_path = os.path.join(dataset_path, split_name)

        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            class_cardinality = len(os.listdir(class_path))

            res[class_name] = res.get(class_name, 0) + class_cardinality



    return res

def get_patches_attention_weight(mask: Tensor, window_size: int = 16) -> Tensor:
    mask = mask.squeeze(dim=1) if mask.dim() > 3 else mask
    
    num_step: int = mask.shape[1] // window_size
    seq_len: int = num_step**2
    row_count: int = 0

    mask_scores: Tensor = ones(
        mask.shape[0], seq_len, 
        device=mask.device, 
        dtype=torch.float
    )

    for num_patch_x in range(1, num_step + 1):
        for num_patch_y in range(1, num_step + 1):
            patch_mask: Tensor = mask[
                :, 
                (num_patch_x - 1) * window_size: num_patch_x * window_size,
                (num_patch_y - 1) * window_size: num_patch_y * window_size
            ]

            patch_mask = patch_mask.flatten(start_dim=-2, end_dim=-1)
            foreground_mask: Tensor = torch.all(patch_mask == 1, dim=-1)
            background_mask: Tensor = torch.all(patch_mask == 0, dim=-1)
            contourn_mask: Tensor = ~foreground_mask & ~background_mask

            mask_scores[background_mask, row_count] = 0.
            mask_scores[foreground_mask, row_count] = 1.
            mask_scores[contourn_mask, row_count] = 2.

            row_count += 1

    cls_mask: Tensor = torch.ones(
        mask_scores.shape[0], 1, 
        device=mask_scores.device, 
        dtype=torch.float
    )

    return torch.cat(
        [
            cls_mask,
            mask_scores # Should be of size batch_size x seq_len 
        ], 
        dim=1
    )


def load_image(path: str) -> Image.Image:
    # Assume 'RGBA' and make sure the background is full of 0s (relevant for CLEOPATRA)
    image: Image.Image = Image.open(path)
    # If image has no alpha, create one (255 everywhere)
    if image.mode == "RGB":
        r, g, b = image.split()
        alpha: Image = Image.new("L", image.size, 255)
        image: Image = Image.merge("RGBA", (r, g, b, alpha))

    # Now image is guaranteed to be RGBA
    r, g, b, alpha = image.split()

    # Create RGB background for CLEOPATRA
    rgb: Image = Image.new('RGB', image.size, (0, 0, 0))
    rgb.paste(image, mask=alpha)
    image: Image = rgb

    # Pad to square
    width, height = image.size
    if width != height:
        if width > height:
            pad = (0, (width - height) // 2, 0, (width - height) // 2)
        else:
            pad = ((height - width) // 2, 0, (height - width) // 2, 0)

        image = T.functional.pad(image, pad)
        alpha = T.functional.pad(alpha, pad)

    # Re-add alpha channel
    image.putalpha(alpha)

    return image

def get_attention_mask(mask: Tensor, window_size: int = 16, use_countourn: bool = True) -> Tensor:
    img_weight: Tensor = get_patches_attention_weight(
        mask=mask,
        window_size=window_size
    )

    foreground_mask: Tensor = (img_weight == 1.)
    contourn_mask: Tensor = (img_weight == 2.)
    attention_mask: Tensor = foreground_mask

    if use_countourn:
        attention_mask |= contourn_mask

    attention_mask = attention_mask.unsqueeze(dim=-1) * attention_mask.unsqueeze(dim=1)
    return attention_mask

class GNNType(Enum):
   GCN = GCN
   GRAPHSAGE = GraphSAGE
   GIN = GIN
   GAT = GAT

class GraphGenout(NamedTuple):
    graph_batch: list[Data]
    avg_cosine_sim: Tensor
    std_cosine_sim: Tensor
    graph_edges_cardinality: Tensor | None
    graph_density: Tensor | None

def generate_sub_edge_index(adjacencies: Tensor, x: Tensor) -> list[Data]:
    data_list = []
    for b in range(adjacencies.size(0)):
        edge_index, _ = dense_to_sparse(adjacencies[b])
        data = Data(
            x=x[b],
            edge_index=edge_index
        )
        data_list.append(data)

    return data_list

def generate_connection(
    patches_emb: Tensor, 
    load_param: float,
    interval_mode: int = 0,
    device: str = "cuda"
) -> GraphGenout:
    patches_emb = patches_emb / patches_emb.norm(dim=-1, keepdim=True)
    cosine_similarities: Tensor = bmm(
        input=patches_emb, 
        mat2=patches_emb.transpose(dim0=1, dim1=2)
    )

    cos_sim_sum: Tensor = cosine_similarities.sum(dim=[1, 2], keepdim=True)
    cos_sim_sum = cos_sim_sum.squeeze(dim=-1)
    avg_cosine_sim: Tensor = cos_sim_sum / (cosine_similarities.shape[1]**2)

    samples: Tensor = cosine_similarities.flatten(start_dim=1)
    std_cosine_sim: Tensor = (
        (samples - avg_cosine_sim)**2
    ).mean(dim=1, keepdim=True).sqrt()

    avg_cosine_sim = avg_cosine_sim.view(-1, 1, 1)
    std_cosine_sim = std_cosine_sim.view(-1, 1, 1)
    
    if interval_mode == 0:
        edge_mask: Tensor = (
            avg_cosine_sim - std_cosine_sim * load_param 
            <= cosine_similarities
        ) & (
            cosine_similarities
            <= avg_cosine_sim + std_cosine_sim * load_param
        )
    elif interval_mode == 1: 
        edge_mask: Tensor = (
            cosine_similarities
            < avg_cosine_sim - std_cosine_sim * load_param 
        ) | (
            cosine_similarities
            > avg_cosine_sim + std_cosine_sim * load_param
        )
    else: 
       edge_mask: Tensor = (
            cosine_similarities
            > avg_cosine_sim + std_cosine_sim * load_param 
        )
    
    diagonal_mask = torch.tensor(
        [x for x in range(edge_mask.shape[1])],
        device=device
    )

    adjcency: Tensor = edge_mask.float()
    adjcency[:, diagonal_mask, diagonal_mask] = 0.

    graph_batch: Batch = generate_sub_edge_index(
        adjacencies=adjcency,
        x=patches_emb
    )

    return GraphGenout(
        graph_batch=graph_batch,
        avg_cosine_sim=avg_cosine_sim,
        std_cosine_sim=std_cosine_sim
    )


def get_epoch_per_style(label: Tensor) -> Tensor: # label is size (batch_size,) 
    styles, epochs = get_style_labels()
    result = []
    for l in label:
        style_name = styles[l.item()]
        epoch_name = style_mapping[style_name]
        epoch_idx = epochs.index(epoch_name)
        result.append(epoch_idx)

    return torch.tensor(result, device=label.device)


"""
Enumeration of available backbone architectures for our ViT-based approach.
Both models (ViT-16 and DeiT-16) are pretrained on ImageNet.
"""
class BackboneType(Enum):
    VIT_16 = "vit_base_patch16_224"
    DEIT_16 = "deit_small_patch16_224"

"""
Enumeration of classification head types for the ViT model.

* CLS_SINGLE: Uses only the [CLS] token as input to the classification head,
  applying it as a multiplicative factor to the learnable parameters.

* SEQ_ENSEMBLE: Applies the learnable matrix to all tokens in the sequence,
  and then aggregates the results for classification.

* SEQ_ENSEMBLE_CLS: Applies the learnable matrix to all tokens in the sequence,
  plus the cls and then aggregates the results for classification.
  """
class HeadType(Enum):
  CLS_SINGLE = "token"
  SEQ_ENSEMBLE = "avg"
  SEQ_ENSEMBLE_MAX = "avgmax"
  SEQ_ENSEMBLE_CLS = "map"
  NONE = ""


class CleopatraInput(NamedTuple):
  image: Tensor
  mask: Tensor
  label: Tensor

class CleopatraEnsembleInput(NamedTuple):
  image: list[Tensor] | Tensor
  label: Tensor
  mask: Tensor | None
  name: list[str] | None

class CleopatraOut(NamedTuple):
  loss: Tensor
  logits: Tensor
  prediction: Tensor
  label: Tensor

class CleopatraMultitaskOut(NamedTuple):
    logits: list[Tensor]


def make_metrics(num_classes: int):
    return MetricCollection({
        "acc": tm.Accuracy(task="multiclass", num_classes=num_classes),
        "f1": tm.F1Score(task="multiclass", num_classes=num_classes, average="macro"),
        "auc": tm.AUROC(task="multiclass", num_classes=num_classes, average="macro"),
    })

def load_from_image_to_tensor(img_path: str) -> tuple[Tensor, Tensor]:
    rgba = load_image(img_path)
    rgba = eval_transform()(rgba)  # Apply transform on full RGBA
    image = rgba[:3]             # [3, 224, 224]
    alpha = (rgba[3] > 0).unsqueeze(0)  # [1, 224, 224]

    return image, alpha


def pairwise_kl(
    logits: Tensor, 
    symmetric: bool = False, 
    reduction: str = "sum", 
    weight: Tensor | None = None
) -> Tensor:
    
    if weight is None:
        weight = torch.ones(
            logits.shape[-1], 
            device=logits.device
        )
    
    red_f = torch.sum if reduction == "sum" else torch.mean
    
    log_prob: Tensor = F.log_softmax(logits, dim=1)       # (B, D)
    prob: Tensor = log_prob.exp()                         # (B, D)

    log_left: Tensor = log_prob.unsqueeze(1)              # (B, 1, D)
    log_right: Tensor = log_prob.unsqueeze(0)             # (1, B, D)

    p_left: Tensor = prob.unsqueeze(1)                    # (B, 1, D)
    p_right: Tensor = prob.unsqueeze(0)                   # (1, B, D)

    kl: Tensor = red_f(
        input=(p_left * (log_left - log_right)) * weight, 
        dim=2
    )  # KL(P_i||P_j)

    if symmetric:
        kl_rev: Tensor = red_f(
            input=(p_right * (log_right - log_left)) * weight, 
            dim=2
        )  # KL(P_j||P_i)
        return kl + kl_rev

    return kl


def kl_similarity(
    logits: Tensor, 
    weight: Tensor | None = None,
    symmetric: bool = False, 
    reduction: str = "sum",
    temperature: float = 6.
) -> Tensor:
    kl_div: Tensor = pairwise_kl(
        logits=logits, 
        symmetric=symmetric, 
        reduction=reduction, 
        weight=weight
    )

    kl_div = torch.div(
        kl_div, 
        temperature
    )

    return torch.exp(-kl_div)   


class CosineSimBundle(NamedTuple):
    cosine_similarity: Tensor
    avg_cosine_sim: Tensor
    std_cosine_sim: Tensor


def get_cosine_stats(
    patches_emb: Tensor,
    temperature: nn.Parameter,
    valid_patch_mask: Tensor | None = None
) -> CosineSimBundle:
    
    if valid_patch_mask is not None:
        patches_emb = patches_emb * valid_patch_mask.unsqueeze(dim=-1)
        non_valid_cardinality: Tensor = (1 - valid_patch_mask).sum(dim=1)
    else: 
        non_valid_cardinality: float = .0

    patches_emb = patches_emb / (patches_emb.norm(dim=-1, keepdim=True) + 1e-5)
    cosine_similarity: Tensor = bmm(
        input=patches_emb, 
        mat2=patches_emb.transpose(dim0=1, dim1=2)
    )

    cos_identity: Tensor = torch.eye(
        cosine_similarity.shape[1], 
        cosine_similarity.shape[2], 
        device=cosine_similarity.device,
        dtype=cosine_similarity.dtype
    )
    cosine_similarity *= (1 - cos_identity) 
    cosine_similarity /= temperature

    cos_sim_sum: Tensor = cosine_similarity.sum(dim=[1, 2])

    cos_cardinality: Tensor = (cosine_similarity.shape[1])**2 - (cosine_similarity.shape[1] + 2 * non_valid_cardinality) 
    avg_cosine_sim: Tensor = cos_sim_sum / (cos_cardinality + 1e-5)
    avg_cosine_sim = avg_cosine_sim.unsqueeze(dim=-1)

    samples: Tensor = cosine_similarity.flatten(start_dim=1)
    samples_identity: Tensor = cos_identity.flatten(start_dim=0)
    samples_agg: Tensor = (samples - avg_cosine_sim) * (1 - samples_identity)

    var_cardinality: Tensor =  samples_agg.shape[-1]**2 - (samples_agg.shape[-1] + 2 * non_valid_cardinality)
    var_cosine_sim: Tensor = (
        samples_agg**2
    ).sum(dim=1) / (var_cardinality + 1e-5)

    std_cosine_sim: Tensor = var_cosine_sim.sqrt()

    avg_cosine_sim = avg_cosine_sim.view(-1, 1, 1)
    std_cosine_sim = std_cosine_sim.view(-1, 1, 1)

    return CosineSimBundle(
        cosine_similarity=cosine_similarity,
        avg_cosine_sim=avg_cosine_sim,
        std_cosine_sim=std_cosine_sim
    )

def add_central_nodes_connection(
    edge_mask: Tensor, 
    num_other_expert: int = 2, 
    agg_nodes_id: list[int] | None = None 
) -> Tensor:
    
    suffix_row: Tensor = torch.zeros(
        edge_mask.shape[0], 
        num_other_expert + 1,
        edge_mask.shape[-1],
        device=edge_mask.device,
        dtype=edge_mask.dtype
    )
    suffix_col: Tensor = torch.zeros(
        edge_mask.shape[0], 
        edge_mask.shape[-1] + num_other_expert + 1,
        num_other_expert + 1,
        device=edge_mask.device,
        dtype=edge_mask.dtype
    )

    edge_mask = torch.cat([
            edge_mask,
            suffix_row
        ], 
        dim=1
    )

    edge_mask = torch.cat([
            edge_mask,
            suffix_col
        ], 
        dim=2
    )

    central_node_id: int = edge_mask.shape[-1] - 1

    if agg_nodes_id is None: 
        agg_nodes_id = list(range(central_node_id - num_other_expert, central_node_id))
        agg_nodes_id = [0] + agg_nodes_id

    # Connect the three aggregation nodes 
    # (Vit, masked vit, extrapolated vit) with the central node
    edge_mask[:, central_node_id, agg_nodes_id] = True
    edge_mask[:, agg_nodes_id, central_node_id] = True

    # Connect the three aggregation nodes 
    # with eachother 
    for agg_id in agg_nodes_id: 
        edge_mask[:, agg_id, agg_nodes_id] = True
        edge_mask[:, agg_nodes_id, agg_id] = True

    return edge_mask

def compute_graph_stats(adjacency: Tensor, valid_patch_mask: Tensor | None) -> tuple[Tensor, Tensor]:
    if valid_patch_mask is not None:
        nodes_cardinality: Tensor = adjacency.shape[1] - (1 - valid_patch_mask).sum(dim=1)
    else:
        nodes_cardinality: Tensor = adjacency.shape[1]

    max_num_edges: Tensor =  0.5 * (nodes_cardinality * (nodes_cardinality - 1))
    num_edges: Tensor = adjacency.sum(dim=[1, 2]) / 2

    return num_edges / max_num_edges, num_edges

def get_raw_edge_mask(
    patches_emb: Tensor, 
    temperature: Tensor, 
    load_param: float,
    adapt_load_param: bool = False,
    valid_patch_mask: Tensor | None = None, 
    mode: Literal["center", "upper"] = "center", 
    threshold: float = 0.7
) -> tuple[Tensor, Tensor, Tensor , Tensor]:
    cosine_similarity, avg_cosine_sim, std_cosine_sim = get_cosine_stats(
        patches_emb=patches_emb, 
        temperature=temperature, 
        valid_patch_mask=valid_patch_mask.float() if valid_patch_mask is not None else valid_patch_mask
    )

    if adapt_load_param:
        load_param /= temperature
    
    if mode == "center":
        edge_mask: Tensor = (
            avg_cosine_sim - std_cosine_sim * load_param 
            <= cosine_similarity
        ) & (
            cosine_similarity
            <= avg_cosine_sim + std_cosine_sim * load_param
        ) 
    else:
        edge_mask: Tensor = cosine_similarity > threshold

    if valid_patch_mask is not None:
        edge_mask &= ( # Removing unvalid tokens from the rows 
            valid_patch_mask.unsqueeze(dim=-1).bool()
        )

        edge_mask &= ( # Removing unvalid tokens from the cols 
            valid_patch_mask.unsqueeze(dim=1).bool()
        )

    return edge_mask, cosine_similarity, avg_cosine_sim, std_cosine_sim

def generate_connection_discrete(
    patches_emb: Tensor, 
    other_global_nodes: Tensor,
    central_node_mode: Literal["mean", "zero"],
    load_param: float,
    temperature: nn.Parameter,
    valid_patch_mask: Tensor | None = None,
    device: str = "cuda", 
    adapt_load_param: bool = False, 
    edge_creation_mode: Literal["center", "upper"] = "center",
    threshold: float = 0.7
) -> GraphGenout:
    
    global_nodes: Tensor = torch.cat([patches_emb, other_global_nodes], dim=1)
    aggregation_nodes: Tensor = torch.cat([
            patches_emb[:, 0, :].unsqueeze(dim=1), 
            other_global_nodes
        ], 
        dim=1
    )
    
    if central_node_mode == "mean":
        central_node: Tensor = aggregation_nodes.mean(dim=1, keepdims=True)
    else:
        central_node: Tensor = torch.zeros_like(patches_emb[:, 0, :])

    if central_node.dim() == 2:
        central_node = central_node.unsqueeze(dim=1)
        
    global_nodes = torch.cat([global_nodes, central_node], dim=1)

    edge_mask, _, avg_cosine_sim, std_cosine_sim = get_raw_edge_mask(
        patches_emb=patches_emb, 
        temperature=temperature,
        valid_patch_mask=valid_patch_mask,
        adapt_load_param=adapt_load_param, 
        load_param=load_param, 
        mode=edge_creation_mode,
        threshold=threshold
    )

    edge_mask = add_central_nodes_connection(edge_mask=edge_mask)
    
    diagonal_mask = torch.tensor(
        [x for x in range(edge_mask.shape[1])],
        device=device
    )

    adjacency: Tensor = edge_mask.float()
    adjacency[:, diagonal_mask, diagonal_mask] = 0.

    graph_batch: list[Data] = generate_sub_edge_index(
        adjacencies=adjacency,
        x=global_nodes
    )

    grap_density, graph_card = compute_graph_stats(
        adjacency=adjacency, 
        valid_patch_mask=valid_patch_mask.float()
    )

    return GraphGenout(
        graph_batch=graph_batch,
        avg_cosine_sim=avg_cosine_sim.view(-1),
        std_cosine_sim=std_cosine_sim.view(-1), 
        graph_density=grap_density,
        graph_edges_cardinality=graph_card
    )

def multiple_generate_connection_discrete(
    patches_emb: list[Tensor], 
    load_param: float,
    temperature: nn.Parameter,
    central_node_mode: Literal["mean", "zero"],
    valid_patch_mask: Tensor | None = None,
    device: str = "cuda", 
    adapt_load_param: bool = False,
    mask_on_learner: int = 2, 
    edge_creation_mode: Literal["center", "upper"] = "center",
    threshold: float = 0.
) -> GraphGenout:
    
    global_nodes: Tensor = torch.cat(patches_emb, dim=1)
    aggregation_nodes: Tensor = global_nodes[:, 0::patches_emb[0].shape[1]]
    
    if central_node_mode == "mean":
        central_node: Tensor = aggregation_nodes.mean(dim=1, keepdims=True)
    else:
        central_node: Tensor = torch.zeros_like(patches_emb[0][:, 0, :])

    if central_node.dim() == 2:
        central_node = central_node.unsqueeze(dim=1)
    global_nodes = torch.cat([global_nodes, central_node], dim=1)

    edge_masks: list[Tensor] = []
    avg_cos_sims: list[Tensor] = []
    std_cos_sims: list[Tensor] = []
    for idx, patch_emb in enumerate(patches_emb):
        if idx == mask_on_learner:
            mask = valid_patch_mask
        else:
            mask = None

        edge_mask, _, avg_cosine_sim, std_cosine_sim = get_raw_edge_mask(
            patches_emb=patch_emb, 
            temperature=temperature,
            valid_patch_mask=mask,
            adapt_load_param=adapt_load_param, 
            load_param=load_param, 
            mode=edge_creation_mode,
            threshold=threshold
        )

        edge_masks.append(edge_mask)
        avg_cos_sims.append(avg_cosine_sim.view(-1))
        std_cos_sims.append(std_cosine_sim.view(-1))

    edge_mask: Tensor = unify_edge_mask(edge_masks=edge_masks)

    edge_mask = add_central_nodes_connection(
        edge_mask=edge_mask, 
        num_other_expert=0, 
        agg_nodes_id=list(range(0, global_nodes.shape[1], patches_emb[0].shape[1]))
    )
    
    diagonal_mask = torch.tensor(
        [x for x in range(edge_mask.shape[1])],
        device=device
    )

    adjacency: Tensor = edge_mask.float()
    adjacency[:, diagonal_mask, diagonal_mask] = 0.

    graph_batch: list[Data] = generate_sub_edge_index(
        adjacencies=adjacency,
        x=global_nodes
    )

    grap_density, graph_card = compute_graph_stats(
        adjacency=adjacency, 
        valid_patch_mask=valid_patch_mask.float()
    )

    return GraphGenout(
        graph_batch=graph_batch,
        avg_cosine_sim=torch.stack(avg_cos_sims, dim=1),
        std_cosine_sim=torch.stack(std_cos_sims, dim=1), 
        graph_density=grap_density,
        graph_edges_cardinality=graph_card
    )

def get_least_idx(ensamble_prediction_t: Tensor, most_used_values: Tensor) -> Tensor:

    least_used_map: Tensor = ensamble_prediction_t != most_used_values.unsqueeze(dim=-1)

    # reverse order of each row and then argmax to get the first True
    rev_idx = torch.flip(least_used_map, dims=[1]).int().argmax(dim=1)
    # convert back the reversed index to its real position
    last_idx = least_used_map.size(1) - 1 - rev_idx

    # assign the last learner to the rows with no True
    no_true = ~least_used_map.any(dim=1)
    last_idx[no_true] = ensamble_prediction_t.shape[1] - 1

    return last_idx

def get_basked_representation(
    ensemble_logits_t: Tensor, 
    ensemble_patches_t: Tensor,
    choice: Literal["least", "most", "merge"] = "least"
) -> tuple[Tensor, Tensor, Tensor]:
    
    ensemble_logits_t = ensemble_logits_t.argmax(dim=-1) # b x n_models

    most_used_values, chosen_idx = torch.mode(ensemble_logits_t, dim=-1)
    
    if choice == "least":
        chosen_idx = get_least_idx(
            ensamble_prediction_t=ensemble_logits_t, 
            most_used_values=most_used_values
        )

    if choice not in ["least", "most"]:
        raise ValueError(f"!!! Mod yet to be developed !!!")

        
    nodes_ids: Tensor = torch.tensor(
        data=[
            x for x in range(ensemble_logits_t.shape[1])
        ], 
        device=ensemble_patches_t.device
    ) 

    chosen_nodes_ids = nodes_ids == chosen_idx.unsqueeze(dim=-1)
    
    final_patches = ensemble_patches_t[chosen_nodes_ids]
    other_global_nodes = ensemble_patches_t[~chosen_nodes_ids]

    other_global_nodes = other_global_nodes.view(
        ensemble_patches_t.shape[0], 
        ensemble_patches_t.shape[1] - 1, 
        ensemble_patches_t.shape[2], -1
    )

    return final_patches, other_global_nodes, chosen_idx

def unify_edge_mask(edge_masks: list[Tensor]) -> Tensor:
    final_edge_mask: list[Tensor] = [edge_masks[0]]

    for idx in range(1, len(edge_masks)):
        final_edge_mask.append(
            torch.cat(
                [torch.ones_like(edge_masks[idk]) for idk in range(idx)]
                +
                [edge_masks[idx]], 
                dim=2
            )
        ) 

        for idj in range(len(final_edge_mask) - 1):
            final_edge_mask[idj] = torch.cat([
                final_edge_mask[idj], 
                torch.ones_like(edge_masks[idj])
            ], dim=2)
           

    return torch.cat(
        final_edge_mask, 
        dim=1
    )


class EnsembleForwardOut(NamedTuple):
    ensemble_logits: Tensor
    learners_logits: Tensor
    additional_log: dict | None = None

class LearnerForwardOut(NamedTuple):
    learners_logits: list[Tensor] | Tensor
    learners_embedding: list[Tensor] | Tensor

class ActFunEnum(Enum):
    RELU = nn.ReLU
    GELU = nn.GELU
    TANH = nn.Tanh
    SIGMOID = nn.Sigmoid


def get_style_labels(path: str = "dataset", path_epoch: str = "dataset_epoch") -> list[str]:
    return os.listdir(os.path.join(path, "train")), os.listdir(os.path.join(path_epoch, "train"))