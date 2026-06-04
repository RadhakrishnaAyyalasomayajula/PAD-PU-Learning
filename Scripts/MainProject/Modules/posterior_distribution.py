import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_posterior_samples(paths, node_index):
    """Loads posterior samples for a given node from multiple model paths."""
    sample_list = []
    for path in paths:
        posterior_path = Path(path) / "FinalPredictions" / "posterior_samples.npy"
        if posterior_path.exists():
            samples = np.load(posterior_path)
            sample_list.append(samples[:, node_index, 1])
        else:
            print(f"⚠️ Missing: {posterior_path}")
    return sample_list

def plot_posterior_distribution(sample_list, node_index=None, title="Posterior Distribution", save_path=None):
    """Plots combined posterior distribution for a given node."""
    if not sample_list:
        print("⚠️ No posterior samples to plot.")
        return

    combined = np.concatenate(sample_list)
    plt.figure(figsize=(6, 4))
    sns.histplot(combined, bins=30, kde=True, color='skyblue')
    plt.title(title)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()