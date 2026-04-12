import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import pytorch_lightning as pl
import numpy as np
import os
from glob import glob
from dataset_handler.sampler import FixedBalancedBatchSampler, create_balanced_batches
from utility.patch_shap_bpt import BPT
from utility.tree_operation import get_adjacency_from_BPT
from utility.utility import BptEnsembleInput, BptEnsembleDatasetPre, CleopatraEnsembleInput, eval_transform, get_attention_mask, load_image, load_json, train_transform
from torch import Tensor
from PIL.Image import Image

class StyleDataset(Dataset):
    """
    Dataset class for multi-style dataset

    Args:
    -----
    `paths` (list of str):
        list of paths to images
    `labels` (list of int):
        list of labels for each image
    `is_train` (bool):
        whether the dataset is for training (True) or testing (False)
    `n_channels` (int):
        number of channels in the images (3 for RGB, 4 for RGBA)
    """
    
    def __init__(
        self, 
        paths: list[str], 
        labels: list[str], 
        is_train: bool, 
        return_name: bool = True
    ) -> None:
        self.paths = paths
        unique_labels = np.unique(labels)
        self.n_styles = len(unique_labels)
        if 0 not in unique_labels:
            labels = [l - 1 for l in labels] # 1-indexed to 0-indexed
        self.labels = labels
        self.transform = train_transform(False) if is_train else eval_transform()
        self.return_name: bool = return_name

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> CleopatraEnsembleInput:
        rgba: Image = load_image(self.paths[idx])
        rgba_t: Tensor = self.transform(rgba)  # Apply transform on full RGBA
        image: Tensor = rgba_t[:3]             # [3, 224, 224]
        alpha: Tensor = (rgba_t[3] > 0).unsqueeze(0)  # [1, 224, 224]
        label: Tensor = torch.tensor(self.labels[idx])

        out: CleopatraEnsembleInput = CleopatraEnsembleInput(
            image=image,
            mask=alpha,
            label=label, 
            name=os.path.basename(self.paths[idx])
        )
            
        return out

class StyleDatasetEnsemble(Dataset):
    """
    Dataset class for multi-style dataset

    Args:
    -----
    `paths` (list of str):
        list of paths to images
    `labels` (list of int):
        list of labels for each image
    `is_train` (bool):
        whether the dataset is for training (True) or testing (False)
    `n_channels` (int):
        number of channels in the images (3 for RGB, 4 for RGBA)
    """
    
    def __init__(
        self, 
        paths: list[list[str]], 
        labels: list[str], 
        is_train: bool, 
        return_name: bool = True
    ):
        self.paths = paths
        unique_labels = np.unique(labels)
        self.n_styles = len(unique_labels)
        if 0 not in unique_labels:
            labels = [l - 1 for l in labels] # 1-indexed to 0-indexed
        self.labels = labels
        self.transform = train_transform() if is_train else eval_transform()
        self.return_name: bool = return_name

    def __len__(self):
        return len(self.paths[0])

    def __getitem__(self, idx: int) -> CleopatraEnsembleInput:

        images: list[Tensor] = []
        alphas: list[Tensor] = []

        for im_pth_lst in self.paths:
            rgba = load_image(im_pth_lst[idx])
            rgba = self.transform(rgba)  # Apply transform on full RGBA
            image = rgba[:3]             # [3, 224, 224]
            alpha = (rgba[3] > 0).unsqueeze(0)  # [1, 224, 224]

            images.append(
                image
            )
            alphas.append(
                alpha
            )

        label: Tensor = torch.tensor(self.labels[idx])
        out: CleopatraEnsembleInput = CleopatraEnsembleInput(
            image=images, 
            mask=alphas,
            label=label
        )

        if self.return_name:
            out.name = os.path.basename(self.paths[0][idx])
        
        return out
    

class StyleDatasetEnsembleBPT(Dataset):
    """
    Dataset class for multi-style dataset

    Args:
    -----
    `paths` (list of str):
        list of paths to images
    `labels` (list of int):
        list of labels for each image
    `is_train` (bool):
        whether the dataset is for training (True) or testing (False)
    `n_channels` (int):
        number of channels in the images (3 for RGB, 4 for RGBA)
    """
    
    def __init__(
        self, 
        img_paths: list[list[str]], 
        bpt_paths: list[list[str]],
        labels: list[int], 
        is_train: bool
    ):
        self.img_paths: list[list[str]] = img_paths
        self.btp_paths: list[list[str]] = bpt_paths

        unique_labels: np.ndarry[int] = np.unique(labels)
        self.n_styles: int = len(unique_labels)

        if 0 not in unique_labels:
            labels = [l - 1 for l in labels] # 1-indexed to 0-indexed

        self.labels: list[int] = labels
        self.transform: T.Compose = train_transform() if is_train else eval_transform()

    def __len__(self) -> int:
        return len(self.img_paths[0])

    def __getitem__(self, idx: int) -> BptEnsembleDatasetPre:

        images: list[Tensor] = []
        alphas: list[Tensor] = []
        bpt_trees: list[BPT] = []

        for im_pth_lst, bpt_pth_lst in zip(self.img_paths, self.btp_paths):
            rgba: Image = load_image(im_pth_lst[idx])
            rgba: Tensor = self.transform(rgba)  # Apply transform on full RGBA
            image: Tensor = rgba[:3]             # [3, 224, 224]
            alpha: Tensor = (rgba[3] > 0).unsqueeze(0)  # [1, 224, 224]

            dict_bpt: dict = load_json(bpt_pth_lst[idx])
            bpt: BPT = BPT.from_dict(dict_bpt)

            images.append(
                image
            )
            alphas.append(
                alpha
            )
            bpt_trees.append(
                bpt
            )

        label: Tensor = torch.tensor(self.labels[idx])
        out: BptEnsembleDatasetPre = BptEnsembleDatasetPre(
            image=images, 
            mask=alphas,
            label=label,
            bpt_info=bpt_trees, 
            name=os.path.basename(self.img_paths[0][idx])
        )
        
        return out
    

def masking_background_collate(
    batch: CleopatraEnsembleInput, 
    use_countourn: bool = False
) -> CleopatraEnsembleInput:
    
    images: list[Tensor] = []
    alphas: list[Tensor] = []
    labels: list[Tensor] = []
    names: list[str] = []

    for img, lbl, alpha, name in batch:
        images.append(img)
        alphas.append(alpha)
        labels.append(lbl)
        names.append(name)

    img_tensor = torch.stack(images)
    alpha_tensor = torch.stack(alphas)
    lbl_tensor = torch.stack(labels)

    attention_weights = get_attention_mask(
        mask=alpha_tensor,
        use_countourn=use_countourn
    )

    return CleopatraEnsembleInput(
        images=img_tensor,
        mask=attention_weights,
        label=lbl_tensor,
        name=names
    )

class MaskingCollate:
    def __init__(self, use_countourn: bool = False):
        self.use_countourn: bool = use_countourn

    def __call__(self, batch) -> CleopatraEnsembleInput:
        return masking_background_collate(
            batch,
            use_countourn=self.use_countourn
        )


def ensemble_collate(
    batch: list[CleopatraEnsembleInput], 
    mask_on_db: int = 0,
    use_countourn: bool = False
) -> CleopatraEnsembleInput:

    alpha_t: list[list[Tensor]] = [[] for _ in range(len(batch))]
    img_t: list[list[Tensor]] = [[] for _ in range(len(batch[0].images))]
    names_t: list[str] = []
    lbl_t: list[Tensor] = []

    for idx, elem in enumerate(batch):
        for db_idx, img in enumerate(elem.images):
            img_t[db_idx].append(img)
            alpha_t[idx].append(elem.alphas[db_idx])

        if elem.name is not None:
            names_t.append(elem.name)

        lbl_t.append(elem.label)


    alpha_tensor: Tensor = torch.stack([alp[mask_on_db] for alp in alpha_t])
    attention_weights = get_attention_mask(
        mask=alpha_tensor.squeeze(dim=1),
        use_countourn=use_countourn
    )

    return CleopatraEnsembleInput(
        image=[torch.stack(imgs) for imgs in img_t],
        label=torch.stack(lbl_t), 
        mask=attention_weights,
        name=names_t
    )

def bpt_ensemble_collate(
    batch: list[BptEnsembleDatasetPre], 
    mask_on_db: int = 0,
    use_countourn: bool = False,
    bpt_percentage: float = 0.3,
    bpt_margin: float = 0.1
) -> BptEnsembleInput:

    alpha_t: list[list[Tensor]] = [[] for _ in range(len(batch))]
    img_t: list[list[Tensor]] = [[] for _ in range(len(batch[0].images))]
    bpt_t: list[list[Tensor]] = [[] for _ in range(len(batch[0].bpt_info))]
    names_t: list[str] = []
    lbl_t: list[Tensor] = []

    for idx, elem in enumerate(batch):
        for db_idx, (img, bpt) in enumerate(zip(elem.images, elem.bpt_info)):
            img_t[db_idx].append(img)
            alpha_t[idx].append(elem.alphas[db_idx])
            bpt_t[db_idx].append(
                get_adjacency_from_BPT(
                    tree=bpt,
                    percentage=bpt_percentage,
                    margin=bpt_margin
                )
            )

        if elem.name is not None:
            names_t.append(elem.name)

        lbl_t.append(elem.label)


    alpha_tensor: Tensor = torch.stack([alp[mask_on_db] for alp in alpha_t])
    attention_weights = get_attention_mask(
        mask=alpha_tensor.squeeze(dim=1),
        use_countourn=use_countourn
    )


    return BptEnsembleInput(
        image=[torch.stack(imgs) for imgs in img_t],
        label=torch.stack(lbl_t), 
        bpt_info=[torch.stack(trees) for trees in bpt_t],
        mask=attention_weights,
        name=names_t
    )


class EnsembleCollate:
    def __init__(
        self, 
        mask_on_db: int = 0,
        use_countourn: bool = True
    ) -> None:
        
        self.use_countourn: bool = use_countourn
        self.mask_on_db: int = mask_on_db

    def __call__(self, batch: list[CleopatraEnsembleInput]) -> CleopatraEnsembleInput:
        return ensemble_collate(
            batch=batch, 
            mask_on_db=self.mask_on_db,
            use_countourn=self.use_countourn
        )
    

class BptEnsembleCollate:
    def __init__(
        self, 
        mask_on_db: int = 0,
        use_countourn: bool = True,
        bpt_percentage: float = 0.3,
        bpt_margin: float = 0.1
    ) -> None:
        
        self.use_countourn: bool = use_countourn
        self.mask_on_db: int = mask_on_db
        self.bpt_percentage: float = bpt_percentage,
        self.bpt_margin: float = bpt_margin

    def __call__(self, batch: list[BptEnsembleDatasetPre]) -> BptEnsembleInput:
        return bpt_ensemble_collate(
            batch=batch, 
            mask_on_db=self.mask_on_db,
            use_countourn=self.use_countourn, 
            bpt_percentage=self.bpt_percentage,
            bpt_margin=self.bpt_margin
        )



class StyleDataModule(pl.LightningDataModule):
    """
    DataModule class for multi-style dataset

    Args:
    -----
    `train_paths` (list of str):
        list of paths to training images
    `train_labels` (list of int):
        list of labels for each training image
    `val_paths` (list of str):
        list of paths to validation images
    `val_labels` (list of int):
        list of labels for each validation image
    `test_paths` (list of str):
        list of paths to testing images
    `test_labels` (list of int):
        list of labels for each testing image
    """
    
    def __init__(
        self, 
        train_paths: list[str], 
        train_labels: list[int], 
        val_paths: list[str], 
        val_labels: list[int], 
        test_paths: list[str], 
        test_labels: list[int],
        batch_size: int, 
        num_workers: int,
        return_name: bool = False,
        train_sampler: bool = False,
        val_sampler: bool = False, 
        masking_vit: bool = False,
        use_countourn: bool = False
    ):
        super().__init__()
        self.train_paths = train_paths
        self.train_labels = train_labels
        self.val_paths = val_paths
        self.val_labels = val_labels
        self.test_paths = test_paths
        self.test_labels = test_labels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.return_name = return_name
        self.masking_vit = masking_vit
        self.use_countourn = use_countourn

        if self.masking_vit:
            self.collate_fn = MaskingCollate(use_countourn=self.use_countourn)

    def train_dataloader(self):
        train_db = StyleDataset(
            self.train_paths, 
            self.train_labels, 
            is_train=True,
            return_name=self.return_name
        )

        if self.train_sampler:
            train_batches = create_balanced_batches(
                self.train_labels, 
                batch_size=self.batch_size, 
                min_per_class=3
            )
            train_sampler = FixedBalancedBatchSampler(train_batches)
            return DataLoader(
                dataset=train_db,
                num_workers=self.num_workers,
                sampler=None,  # don't use sampler and batch_sampler together
                batch_sampler=train_sampler,  # this is your FixedBalancedBatchSampler
            )

        if self.masking_vit:
            return DataLoader(
                dataset=train_db,
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=self.num_workers,
                persistent_workers=True, 
                collate_fn=self.collate_fn
            )
        else:
            return DataLoader(
                dataset=train_db,
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=self.num_workers,
                persistent_workers=True
            )

    def val_dataloader(self):
        val_db = StyleDataset(
            self.val_paths, 
            self.val_labels, 
            is_train=False,
            return_name=self.return_name
        )

        if self.val_sampler:
            val_batches = create_balanced_batches(
                self.val_labels, 
                batch_size=self.batch_size, 
                min_per_class=3
            )
            val_sampler = FixedBalancedBatchSampler(val_batches)
            return DataLoader(
                dataset=val_db,
                num_workers=self.num_workers,
                sampler=None,  # don't use sampler and batch_sampler together
                batch_sampler=val_sampler,  # this is your FixedBalancedBatchSampler
            )

        if self.masking_vit:  
            return DataLoader(
                dataset=val_db,
                batch_size=self.batch_size, 
                num_workers=self.num_workers,
                persistent_workers=True, 
                collate_fn=self.collate_fn
            )
        else:
            return DataLoader(
                dataset=val_db,
                batch_size=self.batch_size, 
                num_workers=self.num_workers,
                persistent_workers=True 
            )
    
    def test_dataloader(self):
        if self.masking_vit:
            return DataLoader(
                StyleDataset(self.test_paths, self.test_labels, is_train=False),
                batch_size=self.batch_size, num_workers=self.num_workers, 
                collate_fn=self.collate_fn
            )
        else:
            return DataLoader(
                StyleDataset(self.test_paths, self.test_labels, is_train=False),
                batch_size=self.batch_size, num_workers=self.num_workers
            )
    

class StyleEnsembleDataModule(pl.LightningDataModule):
    """
    DataModule class for multi-style dataset

    Args:
    -----
    `train_paths` (list of str):
        list of paths to training images
    `train_labels` (list of int):
        list of labels for each training image
    `val_paths` (list of str):
        list of paths to validation images
    `val_labels` (list of int):
        list of labels for each validation image
    `test_paths` (list of str):
        list of paths to testing images
    `test_labels` (list of int):
        list of labels for each testing image
    """
    
    def __init__(
        self, 
        train_paths: list[list[str]], 
        train_labels: list[list[int]], 
        val_paths: list[list[str]], 
        val_labels: list[list[int]], 
        test_paths: list[list[str]], 
        test_labels: list[list[int]],
        batch_size: int, 
        num_workers: int,
        return_name: bool = True,
        masking_vit_on: int = 0,
        use_countourn: bool = False
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "train_paths",
                "train_labels",
                "val_paths",
                "val_labels",
                "test_paths",
                "test_labels"
            ],
            logger=False
        )

        self.train_paths: list[list[str]] = train_paths
        self.train_labels: list[list[int]] = train_labels
        self.val_paths: list[list[str]] = val_paths
        self.val_labels: list[list[int]] = val_labels
        self.test_paths: list[list[str]] = test_paths
        self.test_labels: list[list[int]] = test_labels

        self.collate_fn: EnsembleCollate = EnsembleCollate(
            mask_on_db=masking_vit_on,
            use_countourn=use_countourn
        )

    def train_dataloader(self):
        train_db: StyleDatasetEnsemble = StyleDatasetEnsemble(
            paths=self.train_paths, 
            labels=self.train_labels, 
            is_train=True, 
            return_name=self.hparams.return_name
        )

        return DataLoader(
            dataset=train_db,
            batch_size=self.hparams.batch_size, 
            shuffle=True, 
            num_workers=self.hparams.num_workers,
            persistent_workers=True, 
            collate_fn=self.collate_fn
        )
        

    def val_dataloader(self):
        val_db: StyleDatasetEnsemble = StyleDatasetEnsemble(
            paths=self.val_paths, 
            labels=self.val_labels, 
            is_train=False, 
            return_name=self.hparams.return_name
        )

        return DataLoader(
            dataset=val_db,
            batch_size=self.hparams.batch_size, 
            shuffle=False, 
            num_workers=self.hparams.num_workers,
            persistent_workers=True, 
            collate_fn=self.collate_fn
        ) 
    
    def test_dataloader(self):
        test_db: StyleDatasetEnsemble = StyleDatasetEnsemble(
            paths=self.test_paths, 
            labels=self.test_labels, 
            is_train=False, 
            return_name=self.hparams.return_name
        )

        return DataLoader(
            dataset=test_db,
            batch_size=self.hparams.batch_size, 
            shuffle=False, 
            num_workers=self.hparams.num_workers,
            persistent_workers=True, 
            collate_fn=self.collate_fn
        )
    


class BptStyleEnsembleDataModule(pl.LightningDataModule):
    """
    DataModule class for multi-style dataset

    Args:
    -----
    `train_paths` (list of str):
        list of paths to training images
    `train_labels` (list of int):
        list of labels for each training image
    `val_paths` (list of str):
        list of paths to validation images
    `val_labels` (list of int):
        list of labels for each validation image
    `test_paths` (list of str):
        list of paths to testing images
    `test_labels` (list of int):
        list of labels for each testing image
    """
    
    def __init__(
        self, 
        train_paths_img: list[list[str]], 
        train_paths_bpt: list[list[str]], 
        train_labels: list[list[int]], 
        val_paths_img: list[list[str]], 
        val_paths_bpt: list[list[str]], 
        val_labels: list[list[int]], 
        test_paths_img: list[list[str]], 
        test_paths_bpt: list[list[str]], 
        test_labels: list[list[int]],
        batch_size: int, 
        num_workers: int,
        masking_vit_on: int = 0,
        use_countourn: bool = False,
        bpt_percentage: float = 0.3,
        bpt_margin: float = 0.1
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "train_paths_img",
                "train_labels_img",
                "val_paths_img",
                "val_labels_img",
                "test_paths_img",
                "test_labels_img", 
                "train_paths_bpt", 
                "val_paths_bpt", 
                "test_paths_bpt"
            ],
            logger=False
        )

        self.train_paths_img: list[list[str]] = train_paths_img
        self.train_labels: list[list[int]] = train_labels
        self.val_paths_img: list[list[str]] = val_paths_img
        self.val_labels: list[list[int]] = val_labels
        self.test_paths_img: list[list[str]] = test_paths_img
        self.test_labels: list[list[int]] = test_labels

        self.train_paths_bpt: list[list[str]] = train_paths_bpt
        self.val_paths_bpt: list[list[str]] = val_paths_bpt
        self.test_paths_bpt: list[list[str]] = test_paths_bpt

        self.collate_fn: BptEnsembleCollate = BptEnsembleCollate(
            mask_on_db=masking_vit_on,
            use_countourn=use_countourn, 
            bpt_margin=bpt_margin,
            bpt_percentage=bpt_percentage
        )

    def train_dataloader(self):
        train_db: StyleDatasetEnsembleBPT = StyleDatasetEnsembleBPT(
            paths=self.train_paths_img, 
            labels=self.train_labels,
            bpt_paths=self.train_paths_bpt,
            is_train=True
        )

        return DataLoader(
            dataset=train_db,
            batch_size=self.hparams.batch_size, 
            shuffle=True, 
            num_workers=self.hparams.num_workers,
            persistent_workers=True, 
            collate_fn=self.collate_fn
        )
        

    def val_dataloader(self):
        val_db: StyleDatasetEnsembleBPT = StyleDatasetEnsembleBPT(
            paths=self.val_paths_img, 
            labels=self.val_labels,
            bpt_paths=self.val_paths_bpt,
            is_train=False
        )

        return DataLoader(
            dataset=val_db,
            batch_size=self.hparams.batch_size, 
            shuffle=False, 
            num_workers=self.hparams.num_workers,
            persistent_workers=True, 
            collate_fn=self.collate_fn
        ) 
    
    def test_dataloader(self):
        test_db: StyleDatasetEnsembleBPT = StyleDatasetEnsembleBPT(
            paths=self.test_paths_img, 
            labels=self.test_labels,
            bpt_paths=self.test_paths_bpt,
            is_train=False
        )

        return DataLoader(
            dataset=test_db,
            batch_size=self.hparams.batch_size, 
            shuffle=False, 
            num_workers=self.hparams.num_workers,
            persistent_workers=True, 
            collate_fn=self.collate_fn
        )
    


def load_paths_and_labels(split_dir: str) -> tuple[list[str], list[int]]:

    classes = sorted(os.listdir(split_dir))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    image_paths = []
    labels = []

    for cls_name in classes:
        cls_dir = os.path.join(split_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for f in os.listdir(cls_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):  # include only images
                image_paths.append(os.path.join(cls_dir, f))
                labels.append(class_to_idx[cls_name])
    return image_paths, labels

def load_paths_and_labels_splitted(split_dir) -> dict[str, list[str]]:
    classes = sorted(os.listdir(split_dir))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    image_paths = []
    labels = []
    res: dict[str, list[str]] = {}

    for cls_name in classes:
        cls_dir = os.path.join(split_dir, cls_name)
        res[cls_name] = []
        if not os.path.isdir(cls_dir):
            continue
        for f in os.listdir(cls_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):  # include only images
                res[cls_name].append(os.path.join(cls_dir, f))

    return res

def init_data_module(
    data_dir: str, 
    batch_size: int = 32, 
    num_workers: int = 4, 
    sampler=False, 
    use_test: bool = False, 
    use_masked_vit: bool = False, 
    use_contourn: bool = False,
    return_name: bool = False
):
    train_paths, train_labels = load_paths_and_labels(os.path.join(data_dir, 'train'))
    val_paths, val_labels = load_paths_and_labels(os.path.join(data_dir, 'valid'))
    test_paths, test_labels = load_paths_and_labels(os.path.join(data_dir, 'test'))

    if use_test:
        train_paths += val_paths
        train_labels += val_labels

        return StyleDataModule(
            train_paths=train_paths, 
            train_labels=train_labels,
            val_paths=test_paths, 
            val_labels=test_labels,
            test_paths=test_paths, 
            test_labels=test_labels,
            batch_size=batch_size, 
            num_workers=num_workers,
            train_sampler=sampler, 
            val_sampler=sampler,
            masking_vit=use_masked_vit,
            use_countourn=use_contourn, 
            return_name=return_name
        )
    else:
        return StyleDataModule(
            train_paths=train_paths, 
            train_labels=train_labels,
            val_paths=val_paths, 
            val_labels=val_labels,
            test_paths=test_paths, 
            test_labels=test_labels,
            batch_size=batch_size, 
            num_workers=num_workers,
            train_sampler=sampler, 
            val_sampler=sampler,
            masking_vit=use_masked_vit,
            use_countourn=use_contourn, 
            return_name=return_name
        )
    

def init_data_module_ensemble(
    data_dirs: list[str], 
    batch_size: int = 32, 
    num_workers: int = 4, 
    use_test: bool = False, 
    use_masked_vit_on: int = 0, 
    use_contourn: bool = False
):
    train_dbs: list[list[str]] = []
    val_dbs: list[list[str]] = []
    test_dbs: list[list[str]] = []

    train_lb: list[int] = []
    val_lb: list[int] = []
    test_lb: list[int] = []

    for data_dir in data_dirs: 
        train_paths, train_labels = load_paths_and_labels(os.path.join(data_dir, 'train'))
        val_paths, val_labels = load_paths_and_labels(os.path.join(data_dir, 'valid'))
        test_paths, test_labels = load_paths_and_labels(os.path.join(data_dir, 'test'))

        train_dbs.append(train_paths)
        val_dbs.append(val_paths)
        test_dbs.append(test_paths)

        train_lb = train_labels
        val_lb = val_labels
        test_lb = test_labels

    if use_test:

        for idx in range(len(train_dbs)):
            train_dbs[idx] += val_dbs[idx]

        train_lb += val_lb

        return StyleEnsembleDataModule(
            train_paths=train_dbs, 
            train_labels=train_lb,
            val_paths=test_dbs, 
            val_labels=test_lb,
            test_paths=test_dbs, 
            test_labels=test_lb,
            batch_size=batch_size, 
            num_workers=num_workers,
            masking_vit_on=use_masked_vit_on,
            use_countourn=use_contourn
        )
    else:
        return StyleEnsembleDataModule(
                train_paths=train_dbs, 
                train_labels=train_lb,
                val_paths=val_dbs, 
                val_labels=val_lb,
                test_paths=test_dbs, 
                test_labels=test_lb,
                batch_size=batch_size, 
                num_workers=num_workers,
                masking_vit_on=use_masked_vit_on,
                use_countourn=use_contourn
            )
    

def init_data_module_ensemble_bpt(
    data_dirs_img: list[str], 
    data_dirs_bpt: list[str], 
    batch_size: int = 32, 
    num_workers: int = 4, 
    use_test: bool = False, 
    use_masked_vit_on: int = 0, 
    use_contourn: bool = False
):
    train_dbs_img: list[list[str]] = []
    val_dbs_img: list[list[str]] = []
    test_dbs_img: list[list[str]] = []

    train_dbs_bpt: list[list[str]] = []
    val_dbs_bpt: list[list[str]] = []
    test_dbs_bpt: list[list[str]] = []

    train_lb: list[int] = []
    val_lb: list[int] = []
    test_lb: list[int] = []

    for dir_img, dir_bpt in zip(data_dirs_img, data_dirs_bpt): 
        train_paths_img, train_labels = load_paths_and_labels(os.path.join(dir_img, 'train'))
        val_paths_img, val_labels = load_paths_and_labels(os.path.join(dir_img, 'valid'))
        test_paths_img, test_labels = load_paths_and_labels(os.path.join(dir_img, 'test'))

        train_paths_bpt, _ = load_paths_and_labels(os.path.join(dir_bpt, 'train'))
        val_paths_bpt, _ = load_paths_and_labels(os.path.join(dir_bpt, 'valid'))
        test_paths_bpt, _ = load_paths_and_labels(os.path.join(dir_bpt, 'test'))

        train_dbs_img.append(train_paths_img)
        val_dbs_img.append(val_paths_img)
        test_dbs_img.append(test_paths_img)

        train_dbs_bpt.append(train_paths_bpt)
        val_dbs_bpt.append(val_paths_bpt)
        test_dbs_bpt.append(test_paths_bpt)

        train_lb = train_labels
        val_lb = val_labels
        test_lb = test_labels

    if use_test:

        for idx in range(len(train_dbs_img)):
            train_dbs_img[idx] += val_dbs_img[idx]
            train_dbs_bpt[idx] += val_dbs_bpt[idx]

        train_lb += val_lb

        return BptStyleEnsembleDataModule(
            train_paths_img=train_dbs_img, 
            train_paths_bpt=train_dbs_bpt,
            train_labels=train_lb,
            val_paths_img=test_dbs_img,
            val_paths_bpt=test_dbs_bpt,
            val_labels=test_lb,
            test_paths_img=test_dbs_img, 
            test_paths_bpt=test_dbs_bpt,
            test_labels=test_lb,
            batch_size=batch_size, 
            num_workers=num_workers,
            masking_vit_on=use_masked_vit_on,
            use_countourn=use_contourn
        )
    else:
        return BptStyleEnsembleDataModule(
                train_paths_img=train_dbs_img, 
                train_paths_bpt=train_dbs_bpt,
                train_labels=train_lb,
                val_paths_img=val_dbs_img, 
                val_paths_bpt=val_dbs_bpt,
                val_labels=val_lb,
                test_paths_img=test_dbs_img, 
                test_paths_bpt=test_dbs_bpt,
                test_labels=test_lb,
                batch_size=batch_size, 
                num_workers=num_workers,
                masking_vit_on=use_masked_vit_on,
                use_countourn=use_contourn
            )
    


def init_data_module_augmentation(
    data_dir, 
    batch_size=32, 
    num_workers=4
 ) -> tuple[dict[str, StyleDataModule],dict[str, StyleDataModule]]:
    train_paths: dict[str, list[str]] = load_paths_and_labels_splitted(
        os.path.join(data_dir, 'train')
    )
    val_paths: dict[str, list[str]] = load_paths_and_labels_splitted(
        os.path.join(data_dir, 'valid')
    )

    classes = sorted(os.listdir(os.path.join(data_dir, 'train')))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    return {
        label: StyleDataModule(
            [],[],
            train_paths[label], [class_to_idx[label]] * len(train_paths[label]),
            [], [],
            batch_size=batch_size, 
            num_workers=num_workers, 
            return_name=True
        ) for label in train_paths
    }, {
        label: StyleDataModule(
            [],[],
            val_paths[label], [class_to_idx[label]] * len(val_paths[label]),
            [], [],
            batch_size=batch_size, 
            num_workers=num_workers, 
            return_name=True
        ) for label in val_paths
    }