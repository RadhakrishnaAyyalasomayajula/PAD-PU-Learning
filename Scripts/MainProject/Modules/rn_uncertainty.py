# modules/rn_uncertainty.py

import numpy as np
from scipy.stats import entropy
from sklearn.metrics import pairwise_distances

def agreement_score(rn_sets):
    """
    Computes agreement score across multiple RN selection methods.

    Args:
        rn_sets (List[np.ndarray]): List of arrays with RN indices from each method.

    Returns:
        rn_agreement (dict): Mapping from node index to agreement count.
    """
    from collections import Counter
    flat_rns = np.concatenate(rn_sets)
    counts = Counter(flat_rns)
    return dict(counts)

def entropy_based(embedding_probs, epsilon=1e-9):
    """
    Computes entropy per node based on a softmax-like probability vector.

    Args:
        embedding_probs (np.ndarray): shape (N, D), assumed probability-like per node
        epsilon (float): Small value to avoid log(0)

    Returns:
        entropies (np.ndarray): Shannon entropy for each node
    """
    probs = np.clip(embedding_probs, epsilon, 1.0)
    row_sums = probs.sum(axis=1, keepdims=True)
    norm_probs = probs / row_sums
    entropies = entropy(norm_probs.T)
    return entropies

def distance_based_uncertainty(embeddings, rn_indices, labels, k=5):
    """
    For each RN, compute average distance to k nearest known positives.

    Args:
        embeddings (np.ndarray): All node embeddings (N x D).
        rn_indices (np.ndarray): Indices of reliable negatives.
        labels (np.ndarray): Array of shape (N,) or (N, 1), with 1 for known positives.
        k (int): Number of neighbors.

    Returns:
        distance_scores (np.ndarray): One distance score per RN.
    """
    from sklearn.neighbors import NearestNeighbors

    if len(rn_indices) == 0:
        return np.array([])

    # Flatten labels to 1D if needed
    labels = np.array(labels).flatten()
    known_pos_idx = np.where(labels == 1)[0]

    if len(known_pos_idx) == 0:
        raise ValueError("No known positive examples found.")

    rn_embeddings = embeddings[rn_indices]
    pos_embeddings = embeddings[known_pos_idx]

    nn = NearestNeighbors(n_neighbors=min(k, len(pos_embeddings)))
    nn.fit(pos_embeddings)

    distances, _ = nn.kneighbors(rn_embeddings)
    return distances.mean(axis=1)

