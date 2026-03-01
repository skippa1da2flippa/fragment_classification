import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def plot_explainer_subgraph_with_patches(
    img_path: str,
    explanation,
    graph_batch,
    top_k: int = 10,
    patch_size: int = 16,
    image_size: int = 224,
):
    """
    Plot the important subgraph from GNNExplainer,
    showing each node as its corresponding image patch.
    
    Parameters:
        img_path (str): path to the original image
        explanation: object returned by PyG Explainer
        graph_batch: your Batch object with edge_index and nodes
        top_k (int): how many top nodes to keep
        patch_size (int): size of each ViT patch (typically 16)
        image_size (int): original image resolution (typically 224)
    """

    # -------------------------------------------------------------
    # 1. LOAD IMAGE AND EXTRACT PATCHES
    # -------------------------------------------------------------
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    GRID = image_size // patch_size     # e.g., 224/16 = 14
    num_nodes = GRID * GRID             # 196

    patches = []
    for r in range(GRID):
        for c in range(GRID):
            patch = img[r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size]
            patches.append(patch)

    # -------------------------------------------------------------
    # 2. GET NODE IMPORTANCES
    # -------------------------------------------------------------
    node_mask = explanation.node_mask.detach().cpu()

    # pick top-k nodes
    node_mask = explanation.node_mask.detach().cpu()
    topk_vals, topk_nodes = torch.topk(node_mask.squeeze(dim=1), top_k)


    print(f"Top-K nodes: {topk_nodes}")
    print(f"Scores: {topk_vals.tolist()}")

    # -------------------------------------------------------------
    # 3. FILTER EDGES CONNECTING TOP-K IMPORTANT NODES
    # -------------------------------------------------------------
    edge_index = graph_batch.edge_index.detach().cpu()
    edge_mask = explanation.edge_mask.detach().cpu()

    important_edges = []
    important_weights = []

    for idx, (u, v) in enumerate(edge_index.t().tolist()):
        if u in topk_nodes and v in topk_nodes:
            important_edges.append((u, v))
            important_weights.append(float(edge_mask[idx]))

    print(f"Kept {len(important_edges)} edges inside the top-K subgraph.")

    # -------------------------------------------------------------
    # 4. BUILD NETWORKX SUBGRAPH
    # -------------------------------------------------------------
    G = nx.Graph()

    for n in topk_nodes:
        G.add_node(n, importance=float(node_mask[n]))

    for (u, v), w in zip(important_edges, important_weights):
        G.add_edge(u, v, weight=w)

    # -------------------------------------------------------------
    # 5. PLOT USING NODE IMAGES
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 12))

    # layout can be random or consistent
    pos = nx.spring_layout(G, seed=42)

    # draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        width=[w * 3 for w in important_weights],
        edge_color=important_weights,
        edge_cmap=plt.cm.plasma,
        alpha=0.9
    )

    # draw each node as an image patch
    for n in G.nodes():
        patch = patches[n]                   # retrieve corresponding image patch
        imgbox = OffsetImage(patch, zoom=3)  # enlarge patch for visibility
        ab = AnnotationBbox(
            imgbox,
            pos[n],
            xycoords='data',
            frameon=True,
            bboxprops=dict(edgecolor='black', linewidth=1)
        )
        ax.add_artist(ab)

    # title + style
    plt.title(f"Top-{top_k} Important Nodes & Edges (GNN Explainer)")
    plt.axis("off")
    plt.show()
