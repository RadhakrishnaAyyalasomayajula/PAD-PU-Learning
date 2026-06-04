import pandas as pd
import torch
import numpy as np

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def load_labels(label_csv_path, target_col="PAD"):
    """
    Load binary labels from node CSV and return torch tensor + masks.
    """
    labels_df = pd.read_csv(label_csv_path)[[target_col]].copy()
    labels_df.replace(0, -1, inplace=True)

    labels = labels_df.applymap(lambda x: 1 if x == 1 else -1).values
    labels = torch.tensor(labels, dtype=torch.float)

    known_mask = labels_df.applymap(lambda x: x == 1).any(axis=1).values
    unknown_mask = ~known_mask

    known_indices = np.where(known_mask)[0]
    unknown_indices = np.where(unknown_mask)[0]

    return labels, known_mask, unknown_mask, known_indices, unknown_indices, labels_df


def plot_labelled_embeddings(avg_embeddings, labels_tensor, method="PCA", title=None):
    """
    Visualize embeddings with known vs unknown labels.

    Args:
        avg_embeddings (np.ndarray): Node embeddings (N x D).
        labels_tensor (torch.Tensor): Tensor of labels (1 or -1), shape (N, 1) or (N,).
        method (str): 'PCA' or 'TSNE'.
        title (str): Custom plot title (optional).
    """
    labels = labels_tensor.flatten().numpy()
    known_mask = labels == 1
    unknown_mask = labels == -1

    if method.upper() == "TSNE":
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(labels)//4))
    else:
        reducer = PCA(n_components=2)

    emb_2d = reducer.fit_transform(avg_embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(emb_2d[unknown_mask, 0], emb_2d[unknown_mask, 1],
                c='lightgray', s=10, alpha=0.5, label='Unlabeled')
    plt.scatter(emb_2d[known_mask, 0], emb_2d[known_mask, 1],
                c='red', s=25, alpha=0.8, label='Known PAD (Label=1)')

    plt.legend()
    plt.title(title if title else f"Embedding Visualization ({method.upper()})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

