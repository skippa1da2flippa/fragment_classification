import os
from typing import Type
from tqdm import tqdm
import pytorch_lightning as pl
import torch as to


def graph_augmentation(
    model_type: Type[pl.LightningModule], 
    model_weight_path: str,
    model_hparam_path: str, 
    modules: list[dict[str, pl.LightningDataModule]], 
    out_dataset_path: str = "graph_dataset"
) -> None:
    # Load pretrained model
    vit_embedder = model_type.load_from_checkpoint(
        checkpoint_path=model_weight_path,
        hparams_file=model_hparam_path
    )
    vit_embedder.eval()

    device = to.device("cuda" if to.cuda.is_available() else "cpu")
    vit_embedder.to(device)

    with to.no_grad():
        for idx, split in enumerate(modules):
            split_graph_path = os.path.join(
                out_dataset_path, 
                "train" if idx == 0 else "valid"
            )

            for style, datamodule in split.items():
                if style in ["Byzantine", "Cubism"]:
                    continue
                style_path = os.path.join(split_graph_path, style)
                os.makedirs(style_path, exist_ok=True)
                
                style_res_img: dict[str, to.Tensor] = {}

                for batch in tqdm(
                    datamodule.val_dataloader(),
                    desc=f"Processing {style} ({'train' if idx == 0 else 'valid'})"
                ):
                    imgs, img_names, _ = batch
                    imgs = imgs.to(device)
                    res: to.Tensor = vit_embedder.predict_embedding(imgs).cpu()

                    if isinstance(img_names, (list, tuple)):
                        for name, emb in zip(img_names, res):
                            style_res_img[name] = emb

                # Save embeddings to disk
                for img_name, img_tensor in style_res_img.items():
                    img_dir = os.path.join(style_path, os.path.splitext(img_name)[0])
                    os.makedirs(img_dir, exist_ok=True)
                    image_path = os.path.join(img_dir, "sequence.pt")
                    to.save(img_tensor, image_path)
