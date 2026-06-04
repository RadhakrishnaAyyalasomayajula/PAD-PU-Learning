import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.manifold import TSNE

from embedding_evaluator import analyze_embeddings

# ======================================================
# Paths
# ======================================================
BASE_DIR = Path(
    "/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/"
    "VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Results/All_GNN_Embeddings/"
)

NODE_CSV_PATH = Path(
    "/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/"
    "VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/CuratedData/"
    "LargestComponents/Nodes_with_properties_FC.csv"
)

OUT_DIR = BASE_DIR / "embedding_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR = OUT_DIR / "final_plots"
PLOT_DIR.mkdir(exist_ok=True)

# ======================================================
# Load PAD labels from node CSV
# ======================================================
nodes_df = pd.read_csv(NODE_CSV_PATH)
known_pos_idx = np.where(nodes_df["PAD"] == 1)[0]

# ======================================================
# JSON-safe conversion
# ======================================================
def to_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    else:
        return obj

# ======================================================
# Canonical embedding methods
# ======================================================
methods = {
    "dgi": ["dgi"],
    "gae": ["gae"],
    "vgae": ["vgae"],
    "grace": ["grace"],
    "simgrace": ["simgrace"],
    "mae_random": ["mae_random", "mae-random"],
    "mae_degree": ["mae_degree", "mae-degree"],
    "mae_adaptive": ["mae_adaptive", "mae-adaptive"],
}

def find_file(method_aliases, keyword):
    for f in BASE_DIR.glob("*.npy"):
        fname = f.name.lower()
        if keyword in fname:
            for alias in method_aliases:
                if alias in fname:
                    return f
    return None

# ======================================================
# Run analysis + generate figures
# ======================================================
all_results = {}
stability_summary = {}

for method, aliases in methods.items():
    print(f"▶ Analyzing {method}...")

    avg_path = find_file(aliases, "embeddings")
    std_path = find_file(aliases, "std")

    if avg_path is None:
        raise FileNotFoundError(f"No embeddings file found for {method}")

    avg_embeddings = np.load(avg_path)
    std_embeddings = np.load(std_path) if std_path is not None else None

    # -------- Numeric analysis
    results = analyze_embeddings(
        avg_embeddings=avg_embeddings,
        std_embeddings=std_embeddings,
        method_name=method,
        save_dir=OUT_DIR / method
    )

    all_results[method] = results

    if "stability" in results:
        stability_summary[method] = results["stability"]["mean_node_std"]

    # -------- t-SNE with PAD annotation
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        init="pca"
    )
    emb_2d = tsne.fit_transform(avg_embeddings)

    plt.figure(figsize=(7, 6))
    plt.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c="lightgray",
        s=8,
        alpha=0.4,
        label="Unlabeled"
    )
    plt.scatter(
        emb_2d[known_pos_idx, 0],
        emb_2d[known_pos_idx, 1],
        c="red",
        s=25,
        alpha=0.85,
        label="Known PAD positives"
    )

    plt.title(f"{method.upper()} embeddings (t-SNE)")
    plt.legend()
    plt.axis("off")

    plt.savefig(PLOT_DIR / f"{method}_tsne_pad.png", dpi=300, bbox_inches="tight")
    plt.savefig(PLOT_DIR / f"{method}_tsne_pad.svg", bbox_inches="tight")
    plt.close()

# ======================================================
# Global embedding stability figure (all methods)
# ======================================================
methods_sorted = list(stability_summary.keys())
stability_vals = [stability_summary[m] for m in methods_sorted]

plt.figure(figsize=(8, 4))
plt.bar([m.upper() for m in methods_sorted], stability_vals)
plt.ylabel("Mean node-wise embedding standard deviation")
plt.title("Cross-run embedding stability across methods")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(PLOT_DIR / "embedding_stability_across_methods.png", dpi=300)
plt.savefig(PLOT_DIR / "embedding_stability_across_methods.svg")
plt.close()

# ======================================================
# Save numeric summary
# ======================================================
with open(OUT_DIR / "embedding_summary.json", "w") as f:
    json.dump(to_json_safe(all_results), f, indent=2)

print("✅ Embedding analysis and figure generation complete.")
