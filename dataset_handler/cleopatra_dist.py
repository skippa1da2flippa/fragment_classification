import os
from torch import Tensor, tensor
from utility.utility import get_dataset_cardinality

dataset_path: str = os.path.join("dataset")

def get_dataset_weights(dataset_pth: str = "", full_count: bool = True) -> Tensor:
    
    if dataset_pth == "":
        dataset_pth = dataset_path
    
    card: dict[str, int] = get_dataset_cardinality(
        dataset_path=dataset_path, 
        full_count=full_count
    )

    weights: Tensor = tensor(
        [v for _, v in card.items()]
    )

    weights = weights.sum() / (len(card) * weights)

    return weights

