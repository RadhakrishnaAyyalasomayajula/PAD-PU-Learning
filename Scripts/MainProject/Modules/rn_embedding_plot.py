import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for batch environments
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_rn_embedding_distribution(embeddings, known_pos_idx, rn_idx, method="PCA", title=None):
    """
    Visualize 2D embedding space with:
      - Known Positives (Red)
      - Selected Reliable Negatives (Blue)
      - Rest (Gray)

    Args:
        embeddings (np.ndarray): Node embeddings (N, D)
        known_pos_idx (array-like): Indices of known positive nodes
        rn_idx (array-like): Indices of final selected RNs
        method (str): 'PCA' or 'TSNE'
        title (str): Plot title (optional)
    """
    all_indices = np.arange(len(embeddings))
    other_idx = np.setdiff1d(all_indices, np.concatenate([known_pos_idx, rn_idx]))

    # Reduce to 2D
    if method.upper() == "TSNE":
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//4))
    else:
        reducer = PCA(n_components=2)

    emb_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(emb_2d[other_idx, 0], emb_2d[other_idx, 1],
                c='lightgray', s=10, alpha=0.5, label='Unlabeled')
    plt.scatter(emb_2d[rn_idx, 0], emb_2d[rn_idx, 1],
                c='blue', s=25, alpha=0.7, label='Selected RNs')
    plt.scatter(emb_2d[known_pos_idx, 0], emb_2d[known_pos_idx, 1],
                c='red', s=30, alpha=0.8, label='Known Positives')

    plt.legend()
    plt.title(title if title else f"Embedding Visualization ({method.upper()})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
