import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.stats import normaltest, kurtosis, skew
from pathlib import Path

def analyze_embeddings(avg_embeddings, std_embeddings=None, method_name="Unspecified", save_dir=None):
    """
    Analyzes node embeddings: statistics, clustering, stability, dimensionality.
    
    Args:
        avg_embeddings (np.ndarray): shape (n_nodes, embedding_dim)
        std_embeddings (np.ndarray, optional): same shape, std across runs
        method_name (str): name of the embedding method
        save_dir (str or Path, optional): if provided, saves plots to this directory

    Returns:
        dict: analysis results
    """
    embeddings = avg_embeddings
    n_nodes, embed_dim = embeddings.shape
    flat = embeddings.flatten()

    results = {
        'shape': (n_nodes, embed_dim),
        'embedding_norms': np.linalg.norm(embeddings, axis=1),
        'mean_norm': np.linalg.norm(embeddings, axis=1).mean(),
        'std_norm': np.linalg.norm(embeddings, axis=1).std(),
        'near_zero_dims': np.sum(embeddings.std(axis=0) < 1e-6),
        'dimension_std': embeddings.std(axis=0),
        'normality': {
            'p_value': normaltest(flat)[1],
            'skewness': skew(flat),
            'kurtosis': kurtosis(flat)
        }
    }

    # -------- Similarity Analysis --------
    cos_sim_matrix = cosine_similarity(embeddings)
    cos_sim_off_diag = cos_sim_matrix[np.triu_indices(n_nodes, k=1)]
    results['similarity'] = {
        'mean': cos_sim_off_diag.mean(),
        'std': cos_sim_off_diag.std(),
        'min': cos_sim_off_diag.min(),
        'max': cos_sim_off_diag.max(),
        'identical_pairs': np.sum(cos_sim_off_diag > 0.9999)
    }

    # -------- Clustering (KMeans + Silhouette) --------
    silhouette_scores = []
    k_range = range(2, min(11, n_nodes // 10))
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        silhouette_scores.append(silhouette_score(embeddings, labels))

    best_k = k_range[np.argmax(silhouette_scores)]
    results['clustering'] = {
        'best_k': best_k,
        'best_score': max(silhouette_scores),
        'silhouette_scores': dict(zip(k_range, silhouette_scores))
    }

    # -------- Effective Dimensionality via PCA --------
    pca_full = PCA().fit(embeddings)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    eff_dim_90 = np.argmax(cum_var >= 0.9) + 1
    eff_dim_95 = np.argmax(cum_var >= 0.95) + 1
    eff_dim_99 = np.argmax(cum_var >= 0.99) + 1

    results['dimensionality'] = {
        'effective_90': eff_dim_90,
        'effective_95': eff_dim_95,
        'effective_99': eff_dim_99,
        'dimension_utilization': eff_dim_90 / embed_dim
    }

    # -------- Intrinsic Dimensionality via MLE --------
    nn = NearestNeighbors(n_neighbors=min(10, n_nodes - 1) + 1).fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    distances = distances[:, 1:]

    intrinsic_dims = []
    for row in distances:
        ratios = row[:-1] / row[1:]
        log_ratios = np.log(ratios[ratios > 0])
        if len(log_ratios) > 0:
            intrinsic_dims.append(-1 / np.mean(log_ratios))

    if intrinsic_dims:
        avg_intrinsic = np.mean(intrinsic_dims)
        results['intrinsic'] = {
            'avg_dim': avg_intrinsic,
            'efficiency': avg_intrinsic / embed_dim
        }

    # -------- Concentration --------
    norms = results['embedding_norms']
    results['concentration'] = {
        'norm_cv': norms.std() / norms.mean(),
        'density_variation': distances[:, -1].std() / distances[:, -1].mean()
    }

    # -------- Stability (if std provided) --------
    if std_embeddings is not None:
        mean_std_node = std_embeddings.mean(axis=1)
        mean_std_dim = std_embeddings.mean(axis=0)
        results['stability'] = {
            'mean_node_std': mean_std_node.mean(),
            'max_node_std': mean_std_node.max(),
            'min_node_std': mean_std_node.min(),
            'mean_dim_std': mean_std_dim.mean(),
            'most_stable_nodes': np.argsort(mean_std_node)[:5].tolist(),
            'least_stable_nodes': np.argsort(mean_std_node)[-5:].tolist(),
            'stability_ratio': mean_std_node.max() / max(mean_std_node.min(), 1e-6)
        }

    # -------- Visualization (optional save) --------
    pca_2d = PCA(n_components=2).fit_transform(embeddings)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_nodes // 4))
    tsne_2d = tsne.fit_transform(embeddings)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        axs[0].scatter(pca_2d[:, 0], pca_2d[:, 1], alpha=0.6)
        axs[0].set_title("PCA (2D)")

        axs[1].scatter(tsne_2d[:, 0], tsne_2d[:, 1], alpha=0.6)
        axs[1].set_title("t-SNE (2D)")

        plt.suptitle(f"{method_name} Embeddings: Dimensionality Reduction")
        plt.tight_layout()
        plt.savefig(save_dir / f"{method_name}_projection_plots.png", dpi=300)
        plt.close()

        # Save silhouette plot
        plt.figure()
        plt.plot(list(k_range), silhouette_scores, 'bo-')
        plt.title("Silhouette Score vs Clusters")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / f"{method_name}_silhouette_scores.png", dpi=300)
        plt.close()

    return results