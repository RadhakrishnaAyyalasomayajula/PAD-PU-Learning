# modules/rn_embedding_plotter.py
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for batch environments

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import numpy as np

def plot_rn_embedding_distribution(embeddings, labels_tensor, rn_indices, method="umap", title=None, save_path=None):
    """
    Visualize node embeddings with label info.

    Args:
        embeddings (np.ndarray): Node embeddings (N x D)
        labels_tensor (torch.Tensor or np.ndarray): Labels (1 = known pos, 0 = RN, -1 = unknown)
        rn_indices (np.ndarray): Indices of selected reliable negatives
        method (str): One of 'pca', 'tsne', 'umap'
        title (str): Optional plot title
        save_path (str): If provided, plot will be saved to this path
    """
    labels_np = labels_tensor.flatten().numpy() if hasattr(labels_tensor, 'numpy') else labels_tensor

    known_pos = labels_np == 1
    rn_mask   = np.zeros_like(labels_np, dtype=bool)
    rn_mask[rn_indices] = True
    unknowns = labels_np == -1

    if method.lower() == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//4))
    elif method.lower() == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2)

    emb_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(emb_2d[unknowns, 0], emb_2d[unknowns, 1], c='lightgray', s=10, alpha=0.4, label='Unlabeled')
    plt.scatter(emb_2d[rn_mask, 0], emb_2d[rn_mask, 1], c='red', s=25, alpha=0.8, label='Reliable Negatives')
    plt.scatter(emb_2d[known_pos, 0], emb_2d[known_pos, 1], c='blue', s=25, alpha=0.8, label='Known Positives')

    plt.legend()
    plt.title(title if title else f"Embedding Distribution ({method.upper()})")
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to: {save_path}")
    else:
        print("Warning: No save_path provided. Plot will not be shown in headless mode.")

