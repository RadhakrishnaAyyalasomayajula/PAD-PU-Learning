# scripts/assign_final_novelty_tiers.py

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

# scripts/assign_final_novelty_tiers.py

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

# === Tier assignment logic
def assign_final_tier(row):
    prob = row["mean_probability"]
    epist = row["epistemic_uncertainty"]
    cosine = row["cosine_novelty"]
    mc = row["epistemic_mc_avg"]
    bnn = row["epistemic_bnn_avg"]

    if prob >= 0.8 and epist <= 0.1 and cosine < 1.00:
        return "Tier 1 – Confident & Novel"
    elif prob >= 0.55 and epist > 0.1 and cosine >= 0.75:
        return "Tier 2 – Entropy-based Risk & Novelty"
    elif prob >= 0.8 and epist <= 0.1 and cosine < 0.75:
        return "Tier 3 – Trusted Near-Known"
    elif prob >= 0.7 and mc >= 0.15 and bnn >= 0.15 and cosine >= 0.75:
        return "Tier 4 – MC/BNN High Uncertainty & Novelty"
    else:
        return "Unassigned"

# === Tier caps
TIER_CAPS = {
    "Tier 1 – Confident & Novel": 20,
    "Tier 2 – Entropy-based Risk & Novelty": 12,
    "Tier 3 – Trusted Near-Known": 7,
    "Tier 4 – MC/BNN High Uncertainty & Novelty": 11,
}

# === Tier filtering and selection
def select_top_candidates_by_tier(df: pd.DataFrame):
    df["final_tier"] = df.apply(assign_final_tier, axis=1)

    selected_rows = []
    for tier, cap in TIER_CAPS.items():
        tier_df = df[df["final_tier"] == tier].sort_values("combined_score", ascending=False).head(cap)
        selected_rows.append(tier_df)

    df_top = pd.concat(selected_rows).reset_index(drop=True)
    df_top.to_csv('/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Results/Novel_Positive_Analysis/novel_positive_candidates_tiers.csv', index=False)
    return df_top, df

# === TSNE/UMAP plotting function with tier annotation
def plot_tsne_umap_with_tiers(
    embeddings: np.ndarray,
    known_pos_idx: np.ndarray,
    tiered_df: pd.DataFrame,
    save_dir: Path,
    method: str = "tsne",
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    reducer = TSNE(n_components=2, perplexity=30, random_state=42) if method == "tsne" \
        else umap.UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(embeddings)
    xs, ys = reduced[:, 0], reduced[:, 1]

    all_indices = np.arange(len(xs))
    node_to_coords = dict(zip(all_indices, zip(xs, ys)))
    novel_idx = tiered_df["node_index"].to_numpy()
    background_idx = np.setdiff1d(all_indices, np.union1d(known_pos_idx, novel_idx))

    plt.figure(figsize=(10, 7))
    plt.scatter(xs[background_idx], ys[background_idx], c="lightgray", s=10, alpha=0.3, label="Other nodes")
    plt.scatter(xs[known_pos_idx], ys[known_pos_idx], c="blue", s=40, alpha=0.8, label="Known positives", zorder=2)

    tier_colors = {
        "Tier 1 – Confident & Novel": "darkgreen",
        "Tier 2 – Entropy-based Risk & Novelty": "orangered",
        "Tier 3 – Trusted Near-Known": "goldenrod",
        "Tier 4 – MC/BNN High Uncertainty & Novelty": "darkmagenta",
    }

    for tier, color in tier_colors.items():
        tier_nodes = tiered_df[tiered_df["final_tier"] == tier]["node_index"].values
        coords = np.array([node_to_coords[i] for i in tier_nodes if i in node_to_coords])
        if len(coords):
            plt.scatter(
                coords[:, 0], coords[:, 1],
                c=color, s=60, alpha=0.9, edgecolor="black",
                linewidth=0.5, label=tier, zorder=3
            )

    plt.title(f"{method.upper()} – Known Positives & Novelty Tiers")
    plt.legend()
    plt.axis("off")
    out_path = save_dir / f"{method}_novelty_tiers_plot.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"📍 Saved {method.upper()} plot with novelty tiers → {out_path}")
#
#
#
#
#
#
#
#
#
#
#
#
#
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_tsne_final_sets(
    embeddings: np.ndarray,
    known_pos_idx: np.ndarray,
    results_dir: str | Path,
    save_path: str | Path | None = None,
    perplexity: int = 30,
    random_state: int = 42,
    make_umap: bool = False,   # optional; stays False by default
):
    """
    Plot TSNE with four groups:
      - Known positives (blue)
      - final_cosine_candidates_strict (red)
      - final_mahal_candidates_strict (green)
      - final_prob_heavy_top30 (lightpink)

    Ensures the three candidate sets are mutually exclusive with priority:
      cosine > mahal > prob
    """
    results_dir = Path(results_dir)
    if save_path is None:
        save_path = results_dir / "tsne_final_sets.png"

    # --- Load final selection CSVs ---
    f_cos = results_dir / "final_cosine_candidates_strict.csv"
    f_mah = results_dir / "final_mahal_candidates_strict.csv"
    f_prob = results_dir / "final_prob_heavy_top30.csv"

    for p in [f_cos, f_mah, f_prob]:
        if not p.exists():
            raise FileNotFoundError(f"Missing expected file: {p}")

    df_cos = pd.read_csv(f_cos)
    df_mah = pd.read_csv(f_mah)
    df_prob = pd.read_csv(f_prob)

    # Require node_index for plotting (we map indices to TSNE coords)
    for name, df in [("cosine", df_cos), ("mahal", df_mah), ("prob", df_prob)]:
        if "node_index" not in df.columns:
            raise ValueError(f"'node_index' missing in {name} dataframe ({results_dir})")

    # --- Get index arrays; keep only indices within embeddings range ---
    n = embeddings.shape[0]

    def clean_idx(arr_like):
        arr = np.asarray(arr_like, dtype=int)
        arr = arr[(arr >= 0) & (arr < n)]
        return np.unique(arr)

    idx_cos = clean_idx(df_cos["node_index"].values)
    idx_mah = clean_idx(df_mah["node_index"].values)
    idx_prob = clean_idx(df_prob["node_index"].values)
    known_pos_idx = clean_idx(known_pos_idx)

    # --- Enforce mutual exclusivity with priority: cosine > mahal > prob ---
    # Anything that appears in cosine is removed from mahal & prob; anything
    # that appears in mahal is removed from prob.
    set_cos = set(idx_cos)
    set_mah = set(idx_mah) - set_cos
    set_prob = set(idx_prob) - set_cos - set_mah

    # Report overlaps (before/after)
    overlap_cos_mah = len(set(idx_cos) & set(idx_mah))
    overlap_cos_prob = len(set(idx_cos) & set(idx_prob))
    overlap_mah_prob = len(set(idx_mah) & set(idx_prob))
    if overlap_cos_mah or overlap_cos_prob or overlap_mah_prob:
        print(f"Found overlaps before enforcing exclusivity: "
              f"cos∩mah={overlap_cos_mah}, cos∩prob={overlap_cos_prob}, mah∩prob={overlap_mah_prob}")
        print("Applying priority (cosine > mahal > prob) to make sets disjoint.")

    idx_cosine = np.array(sorted(set_cos), dtype=int)
    idx_mahal  = np.array(sorted(set_mah), dtype=int)
    idx_prob   = np.array(sorted(set_prob), dtype=int)

    # --- Compute TSNE on ALL embeddings (keeps global structure) ---
    reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    reduced = reducer.fit_transform(embeddings)
    xs, ys = reduced[:, 0], reduced[:, 1]

    all_indices = np.arange(n)
    selected_union = np.union1d(np.union1d(idx_cosine, idx_mahal), idx_prob)
    background_idx = np.setdiff1d(all_indices, np.union1d(known_pos_idx, selected_union))

    # --- Plot ---
    plt.figure(figsize=(10, 7))

    # Background
    plt.scatter(xs[background_idx], ys[background_idx], c="lightgray", s=10, alpha=0.25, label="Other nodes", zorder=1)

    # Known positives
    plt.scatter(xs[known_pos_idx], ys[known_pos_idx], c="blue", s=40, alpha=0.8,
                label=f"Known positives ({len(known_pos_idx)})", zorder=2)

    # Cosine strict (red)
    if len(idx_cosine):
        plt.scatter(xs[idx_cosine], ys[idx_cosine], c="red", s=60, alpha=0.9,
                    edgecolor="black", linewidth=0.5,
                    label=f"Cosine strict ({len(idx_cosine)})", zorder=3)

    # Mahalanobis strict (green)
    if len(idx_mahal):
        plt.scatter(xs[idx_mahal], ys[idx_mahal], c="green", s=60, alpha=0.9,
                    edgecolor="black", linewidth=0.5,
                    label=f"Mahalanobis strict ({len(idx_mahal)})", zorder=3)

    # Prob top (lightpink)
    if len(idx_prob):
        plt.scatter(xs[idx_prob], ys[idx_prob], c="lightpink", s=55, alpha=0.9,
                    edgecolor="black", linewidth=0.5,
                    label=f"Prob top ({len(idx_prob)})", zorder=3)

    plt.title("TSNE – Known Positives vs Final Selections")
    plt.legend()
    plt.axis("off")

    save_path = Path(save_path)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"📍 Saved TSNE plot → {save_path}")

    # Also return the (now disjoint) index arrays in case you want to reuse them
    return {
        "idx_cosine": idx_cosine,
        "idx_mahal": idx_mahal,
        "idx_prob": idx_prob,
        "known_pos_idx": known_pos_idx
    }