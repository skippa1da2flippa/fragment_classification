import torch
import numpy as np
import cv2
from torch import Tensor
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.data import Batch

from models_handler.frenziness.gnn import UltimateGraphApproach
from exaplainability.gnn_expl_pipeline import plot_explainer_subgraph_with_patches
from utility.utility import GraphGenout, generate_connection, load_from_image_to_tensor
from models_handler.transformer.vit import VitClassifier

device = 'cuda'
img_path = "dataset\\test\\Egyptian\\Frag_107.png"

PATCH_SIZE = 16
IMG_SIZE = 224
PATCHES_PER_ROW = IMG_SIZE // PATCH_SIZE  # = 14

# -----------------------------
# LOAD MODELS
# -----------------------------
model: UltimateGraphApproach = UltimateGraphApproach.load_from_checkpoint(
    hparams_file="FINAL_GNN_INNER_GAT\\GAT\\version_0\\hparams.yaml",
    checkpoint_path="FINAL_GNN_INNER_GAT\\GAT_wght\\weights.ckpt"
)

backbone: VitClassifier = VitClassifier.load_from_checkpoint(
    hparams_file="final_VIT\\FULL_VIT_TEST_logs\\FINAL_VIT_csv_extrapolated\\version_0\\hparams_extrapolated.yaml",
    checkpoint_path="final_VIT\\FINAL_VIT_CHKT\\weights_extrapolated.ckpt"
)

model.eval()
backbone.eval()

# -----------------------------
# BUILD EXPLAINER
# -----------------------------
explainer = Explainer(
    model=model.gnn,
    algorithm=GNNExplainer(epochs=200, lr=0.01),
    explanation_type='model',
    node_mask_type='object',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs',
    ),
)

# -----------------------------
# LOAD IMAGE → PATCHES → GRAPH
# -----------------------------
batch, _ = load_from_image_to_tensor(img_path=img_path)
batch = batch.unsqueeze(0).detach().to(device)

# patch embeddings
patches: Tensor = backbone.predict_embedding(batch).detach()

# graph construction
graph_out: GraphGenout = generate_connection(
    patches_emb=patches,
    load_param=model.hparams.graph_load_param,
    device=device
)

graph_batch = Batch.from_data_list(graph_out.graph_batch).to(device)
graph_batch.x = graph_batch.x.detach()

index_to_explain = 2

# -----------------------------
# RUN EXPLAINER
# -----------------------------
explanation = explainer(
    graph_batch.x,
    graph_batch.edge_index,
    index=index_to_explain,
)

node_mask = explanation.node_mask  # shape [196, F]
node_importance = node_mask.abs().sum(dim=1)  # [196]

plot_explainer_subgraph_with_patches(
    img_path=img_path,
    explanation=explanation,
    graph_batch=graph_batch,
    top_k=8,              # choose any K
    patch_size=16,
    image_size=224
)

# # --------------------------------
# # MOST IMPORTANT NODES
# # --------------------------------
# k = 5
# topk_vals, topk_idx = torch.topk(node_importance, k)

# print("Top important nodes:", topk_idx.tolist())
# print("Scores:", topk_vals.tolist())

# # --------------------------------
# # BUILD HEATMAP FROM NODE IMPORTANCE
# # --------------------------------
# # load real image
# img = cv2.imread(img_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# import numpy as np
# import cv2

# patch_size = 16   # your patches are 16x16
# H, W = img.shape[:2]
# grid = H // patch_size           # 224/16 = 14

# # Create zero heatmap for all patches
# patch_heatmap = np.zeros((grid * grid,), dtype=float)

# # Fill importance for selected nodes
# for idx, s in zip(topk_idx.tolist(), topk_vals.tolist()):
#     if idx < len(patch_heatmap):
#         patch_heatmap[idx] = s

# # Normalize
# patch_heatmap = patch_heatmap.reshape(grid, grid)
# patch_heatmap = patch_heatmap / (patch_heatmap.max() + 1e-8)

# # Resize patch heatmap → original image resolution (224×224)
# heatmap_resized = cv2.resize(
#     patch_heatmap,
#     (W, H),
#     interpolation=cv2.INTER_LINEAR
# )

# # Convert to uint8 for OpenCV
# heatmap_uint8 = np.uint8(255 * heatmap_resized)

# # Apply colormap (JEt)
# heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

# # Convert image if needed
# if img.dtype != np.uint8:
#     img = (img * 255).astype(np.uint8)

# # Blend heatmap with original image
# alpha = 0.45
# overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

# # Save
# cv2.imwrite("gnn_explanation_overlay.png", overlay)
# print("Saved overlay to gnn_explanation_overlay.png")

# # ------------------------------------------------------------
# #  END OF INSERTED BLOCK
