# modules/rn_selector.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

def spy_method(embeddings, labels, rn_ratio=1.0, spy_ratio=0.15, random_state=42):
    pos_idx = np.where(labels == 1)[0]
    unlabeled_idx = np.where(labels == -1)[0]

    n_spies = max(1, int(spy_ratio * len(pos_idx)))
    np.random.seed(random_state)
    spy_indices = np.random.choice(pos_idx, size=n_spies, replace=False)

    spy_embeddings = embeddings[spy_indices]
    unlabeled_embeddings = embeddings[unlabeled_idx]

    train_X = np.vstack((spy_embeddings, unlabeled_embeddings))
    train_y = np.array([1] * len(spy_embeddings) + [0] * len(unlabeled_embeddings))

    clf = GaussianNB()
    clf.fit(train_X, train_y)
    probs = clf.predict_proba(unlabeled_embeddings)[:, 1]
    spy_probs = clf.predict_proba(spy_embeddings)[:, 1]
    threshold = spy_probs.max()

    reliable_negative_indices = unlabeled_idx[probs < threshold]
    max_rn = int(rn_ratio * len(pos_idx))
    return reliable_negative_indices[:max_rn]

def kmeans_method(embeddings, labels, rn_ratio=1.0, n_clusters=20, random_state=42):
    pos_idx = np.where(labels == 1)[0]
    unlabeled_idx = np.where(labels == -1)[0]

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(embeddings)
    cluster_labels = kmeans.labels_

    pos_clusters = set(cluster_labels[pos_idx])
    candidate_idx = [i for i in unlabeled_idx if cluster_labels[i] not in pos_clusters]

    if not candidate_idx:
        return np.array([])

    dists = pairwise_distances(embeddings[candidate_idx], embeddings[pos_idx])
    avg_dist = dists.mean(axis=1)
    sorted_indices = np.argsort(avg_dist)[::-1]
    max_rn = int(rn_ratio * len(pos_idx))
    return np.array(candidate_idx)[sorted_indices[:max_rn]]

def knn_method(embeddings, labels, rn_ratio=1.0, k=5):
    pos_idx = np.where(labels == 1)[0]
    unlabeled_idx = np.where(labels == -1)[0]

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(embeddings[pos_idx])
    distances, _ = nn.kneighbors(embeddings[unlabeled_idx])
    avg_dist = distances.mean(axis=1)

    sorted_indices = np.argsort(avg_dist)[::-1]
    max_rn = int(rn_ratio * len(pos_idx))
    return unlabeled_idx[sorted_indices[:max_rn]]

def gpu_method(embeddings, labels, rn_ratio=1.0, n_components=1, random_state=42):
    pos_idx = np.where(labels == 1)[0]
    unlabeled_idx = np.where(labels == -1)[0]

    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(embeddings[pos_idx])
    log_probs = gmm.score_samples(embeddings[unlabeled_idx])

    sorted_indices = np.argsort(log_probs)
    max_rn = int(rn_ratio * len(pos_idx))
    return unlabeled_idx[sorted_indices[:max_rn]]

def ccrne_method(embeddings, labels, rn_ratio=1.0, n_clusters=20, random_state=42):
    pos_idx = np.where(labels == 1)[0]
    unlabeled_idx = np.where(labels == -1)[0]

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(embeddings)
    cluster_labels = kmeans.labels_

    pos_clusters = set(cluster_labels[pos_idx])
    candidate_idx = [i for i in unlabeled_idx if cluster_labels[i] not in pos_clusters]

    max_rn = int(rn_ratio * len(pos_idx))
    return np.array(candidate_idx[:max_rn])

def spy_method_ensemble(embeddings, labels, rn_ratio=1.0, spy_ratio=0.15, n_runs=5):
    """
    Run spy_method multiple times with different random seeds to reduce variance.
    
    Returns:
        List of RN index arrays, one per run.
    """
    all_rn_sets = []
    for seed in range(n_runs):
        rns = spy_method(
            embeddings=embeddings,
            labels=labels,
            rn_ratio=rn_ratio,
            spy_ratio=spy_ratio,
            random_state=seed
        )
        all_rn_sets.append(rns)
    return all_rn_sets