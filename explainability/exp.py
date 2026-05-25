import os
import torch
import timm
import cv2
import numpy as np
from PIL import Image

from vit_explain.vit_rollout import VITAttentionRollout
from models_handler.transformer.vit import VitClassifier
from torchvision import transforms
from torchvision.datasets import ImageFolder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def show_mask_on_image(image_pil, mask):
    image = np.array(image_pil.convert("RGB"))
    image = image.astype(np.float32) / 255.0

    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask = mask / (mask.max() + 1e-8)

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255.0
    heatmap = heatmap[..., ::-1]

    cam = heatmap + image
    cam = cam / np.max(cam)

    return np.uint8(255 * cam)


def load_image(image_path, transform, device):
    img = Image.open(image_path).convert("RGBA")
    arr = np.array(img)

    rgb = arr[..., :3].astype(np.float32)
    alpha = arr[..., 3].astype(np.float32) / 255.0

    rgb = rgb * alpha[..., None]
    rgb = rgb.astype(np.uint8)

    image = Image.fromarray(rgb)

    input_tensor = transform(image).unsqueeze(0).to(device)
    return image, input_tensor

def check_vit(
    weights_pth:str = "", 
    hparams_pth:str = "", 
    dataset_pth:str = "",
    output_dir:str = "",
    discard_ratio:float = 0.1,
    head_fusion:str="mean"
):
    os.makedirs(output_dir, exist_ok=True)

    model: VitClassifier = VitClassifier.load_from_checkpoint(
            checkpoint_path=weights_pth,
            hparams_file=hparams_pth
        ).to(device)
    
    model.eval()
    for name, module in model.named_modules():
        if "attn" in name.lower() or "attention" in name.lower():
            print(name, "->", module.__class__.__name__)

    for module in model.modules():
        if hasattr(module, "fused_attn"):
            module.fused_attn = False

    selected = {}

    dataset = ImageFolder(root=dataset_pth, transform=None)


    for image_path, label in dataset.samples:
        if label not in selected:
            selected[label] = image_path

        if len(selected) == len(dataset.classes):
            break

    grad_rollout = VITAttentionRollout(model, discard_ratio=discard_ratio, head_fusion=head_fusion)
    # mask = grad_rollout(input_tensor, category_index=243)

    for true_class_idx, image_path in selected.items():
        true_class_name = dataset.classes[true_class_idx]

        image_pil, input_tensor = load_image(
            image_path,
            val_transform,
            device
        )

        with torch.no_grad():
            logits = model(input_tensor)
            pred_idx = logits.argmax(dim=1).item()
            # print(pred_idx)
            # print(dataset.classes)
            pred_class_name = dataset.classes[pred_idx]

        mask = grad_rollout(
            input_tensor,
            # category_index=pred_idx
        )

        visualization = show_mask_on_image(image_pil, mask)

        save_path = os.path.join(
            output_dir,
            f"{true_class_name}_pred_{pred_class_name}.png"
        )

        Image.fromarray(visualization).save(save_path)

        print(f"Saved: {save_path}")


check_vit(
    weights_pth="C:\\Users\\skippa\\Documents\\projects\\fragment_classification\\FINAL_MODELS\\final_VIT_max_accuracy\\FINAL_VIT_CHKT\\weights_max_acc-v1.ckpt",
    hparams_pth="C:\\Users\\skippa\\Documents\\projects\\fragment_classification\\FINAL_MODELS\\final_VIT_max_accuracy\\FULL_VIT_TEST_logs\\FINAL_VIT_csv_max_acc\\version_1\\hparams.yaml",
    dataset_pth="C:\\Users\\skippa\\Documents\\projects\\fragment_classification\\datasets\\fragment_dataset\\test",
    output_dir="explainability\\fragment_rollout_mean05",
    head_fusion="mean",
    discard_ratio=0.5
)