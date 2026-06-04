# Modules/novel_positive_selector.py
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import torch  # (not used directly here but kept if you import this elsewhere)

from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import LocalOutlierFactor
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.covariance import LedoitWolf

from scipy.spatial import distance
import umap


def _uncertainty_tier(u: float, q1: float, q2: float) -> str:
    if u <= q1:
        return "low"
    elif q1 < u <= q2:
        return "mid"
    else:
        return "high"


def compute_novel_positive_candidates(
    ensemble_csv_path: str | Path,
    embeddings: np.ndarray,
    known_positive_indices: np.ndarray,
    save_dir: str | Path,
    top_k: int = 50,
    min_mean_prob: float = 0.8,
    max_uncertainty: float = 0.4,
    lof_k: int = 20,
    generate_plots: bool = True,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ensemble_csv_path)

    mask = (
        (df["mean_probability"] >= min_mean_prob)
        & (df["epistemic_uncertainty"] <= max_uncertainty)
        & (df["predicted_label"] == 1)
    )

    # (Unused in original logic, kept if you want to switch later)
    mask_epist_both = (
        (df["mean_probability"] >= min_mean_prob)
        & (df["epistemic_mc_avg"] <= 0.2)
        & (df["epistemic_bnn_avg"] <= 0.2)
        & (df["predicted_label"] == 1)
    )

    df_cand = df.loc[mask].copy()
    df_cand = df_cand[~df_cand["node_index"].isin(known_positive_indices)]
    if df_cand.empty:
        print("⚠️  No candidates pass confidence / uncertainty thresholds.")
        return pd.DataFrame()

    cand_idx = df_cand["node_index"].to_numpy()
    known_emb = embeddings[known_positive_indices]
    cand_emb = embeddings[cand_idx]

    print("cand_emb shape:", cand_emb.shape)
    print("known_emb shape:", known_emb.shape)

    # === Cosine novelty (mean distance to all known positives; embeddings normalized)
    known_emb_norm = normalize(known_emb, axis=1)
    cand_emb_norm = normalize(cand_emb, axis=1)
    dist_cos = cosine_distances(cand_emb_norm, known_emb_norm).mean(axis=1)
    df_cand["cosine_novelty"] = dist_cos

    # === LOF novelty (on full space)
    lof = LocalOutlierFactor(
        n_neighbors=min(lof_k, len(df)),
        metric="cosine",
        novelty=True
    ).fit(embeddings)
    lof_scores = -lof.score_samples(cand_emb)
    df_cand["lof_novelty"] = lof_scores

    # === Mahalanobis novelty (distribution-aware; shrinkage covariance for stability)
    # Use raw (unnormalized) embeddings for covariance structure
    cov_estimator = LedoitWolf().fit(known_emb)
    mean_known = cov_estimator.location_
    inv_cov = np.linalg.inv(cov_estimator.covariance_)

    # Compute distance to the KNOWN-POSITIVES MEAN under inv_cov
    # (Vectorized via cdist to a single target row)
    mahal = distance.cdist(cand_emb, mean_known[None, :], metric="mahalanobis", VI=inv_cov).ravel()
    df_cand["mahalanobis_novelty"] = mahal

    # === Quantile thresholds on candidates
    cosine_high = df_cand["cosine_novelty"].quantile(0.66)
    cosine_low = df_cand["cosine_novelty"].quantile(0.33)

    lof_high = df_cand["lof_novelty"].quantile(0.66)
    lof_low = df_cand["lof_novelty"].quantile(0.33)

    mahal_high = df_cand["mahalanobis_novelty"].quantile(0.66)
    mahal_low = df_cand["mahalanobis_novelty"].quantile(0.33)

    uncert_low = df_cand["epistemic_uncertainty"].quantile(0.33)
    uncert_high = df_cand["epistemic_uncertainty"].quantile(0.66)

    prob_high = df_cand["mean_probability"].quantile(0.66)
    prob_low = df_cand["mean_probability"].quantile(0.33)

    # Uncertainty tiers
    df_cand["uncertainty_tier"] = df_cand["epistemic_uncertainty"].apply(
        lambda x: _uncertainty_tier(x, uncert_low, uncert_high)
    )

    # Normalized metrics (0..1 within candidate pool)
    def _minmax(s: pd.Series) -> pd.Series:
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-12)

    df_cand["cosine_norm"] = _minmax(df_cand["cosine_novelty"])
    df_cand["prob_norm"] = _minmax(df_cand["mean_probability"])
    df_cand["mahal_norm"] = _minmax(df_cand["mahalanobis_novelty"])

    # Combined scores
    df_cand["combined_score_cosine"] = 0.90 * df_cand["cosine_norm"] + 0.10 * df_cand["prob_norm"]
    df_cand["combined_score_prob"]   = 0.10 * df_cand["cosine_norm"] + 0.90 * df_cand["prob_norm"]
    df_cand["combined_score_mahal"]  = 0.90 * df_cand["mahal_norm"]  + 0.10 * df_cand["prob_norm"]
    # Fused: cosine + mahal + prob (heaviest on novelty)
    df_cand["combined_score_fused"]  = 0.45 * df_cand["cosine_norm"] + 0.45 * df_cand["mahal_norm"] + 0.10 * df_cand["prob_norm"]

    # === Top-K tables (keep your originals; add optional fused/mahal if you want)
    df_top_cosine = (
        df_cand.sort_values("combined_score_cosine", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    df_top_prob = (
        df_cand.sort_values("combined_score_prob", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    df_top_mahal = (
        df_cand.sort_values("combined_score_mahal", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    df_top_fused = (
        df_cand.sort_values("combined_score_fused", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    # === Tiering (now considers BOTH cosine and Mahalanobis + LOF + uncertainty)
    def assign_tier(row):
        novelty_cos = row["cosine_novelty"]
        novelty_lof = row["lof_novelty"]
        novelty_mah = row["mahalanobis_novelty"]
        uncertainty = row["epistemic_uncertainty"]

        # Strongest: high on BOTH cosine and Mahalanobis, high LOF, and low uncertainty
        if (
            (novelty_cos >= cosine_high)
            and (novelty_mah >= mahal_high)
            and (novelty_lof >= lof_high)
            and (uncertainty <= uncert_low)
        ):
            return "Tier 1"
        # Solid: above lower thresholds on both novelty metrics and low uncertainty
        elif (
            (novelty_cos >= cosine_low)
            and (novelty_mah >= mahal_low)
            and (novelty_lof >= lof_low)
            and (uncertainty <= uncert_low)
        ):
            return "Tier 2"
        # Novel but riskier: high novelties but uncertainty not the lowest
        elif (
            (novelty_cos >= cosine_high)
            and (novelty_mah >= mahal_high)
            and (novelty_lof >= lof_high)
            and (uncertainty > uncert_low)
        ):
            return "Tier 3"
        else:
            return "Tier 4"

    for _df in (df_top_cosine, df_top_prob, df_top_mahal, df_top_fused):
        _df["novelty_tier"] = _df.apply(assign_tier, axis=1)

    # === Save CSVs
    out_csv_cosine = save_dir / "novel_positive_candidates_cosine_9.csv"
    out_csv_prob   = save_dir / "novel_positive_candidates_prob_9.csv"
    out_csv_mahal  = save_dir / "novel_positive_candidates_mahal_9.csv"
    out_csv_fused  = save_dir / "novel_positive_candidates_fused_9.csv"
    out_csv_all    = save_dir / "all_positive_candidates_9.csv"

    df_cand.to_csv(out_csv_all, index=False)
    df_top_cosine.to_csv(out_csv_cosine, index=False)
    df_top_prob.to_csv(out_csv_prob, index=False)
    df_top_mahal.to_csv(out_csv_mahal, index=False)
    df_top_fused.to_csv(out_csv_fused, index=False)

    # === Plots
    if generate_plots:
        colour_map = {"low": "green", "mid": "orange", "high": "red"}
        colors = df_cand["uncertainty_tier"].map(colour_map)

        # Cosine vs probability
        plt.figure(figsize=(8, 6))
        plt.scatter(
            df_cand["cosine_novelty"],
            df_cand["mean_probability"],
            c=colors,
            alpha=0.5,
        )
        for tier, col in colour_map.items():
            plt.scatter([], [], color=col, label=f"{tier}-uncert.")
        plt.xlabel("Cosine Novelty")
        plt.ylabel("Mean Predicted Probability")
        plt.title("Novelty vs Confidence (all candidates) — Cosine")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_dir / "novelty_vs_confidence_cosine.png", bbox_inches="tight")
        plt.close()

        # Mahalanobis vs probability
        plt.figure(figsize=(8, 6))
        plt.scatter(
            df_cand["mahalanobis_novelty"],
            df_cand["mean_probability"],
            c=colors,
            alpha=0.5,
        )
        for tier, col in colour_map.items():
            plt.scatter([], [], color=col, label=f"{tier}-uncert.")
        plt.xlabel("Mahalanobis Novelty")
        plt.ylabel("Mean Predicted Probability")
        plt.title("Novelty vs Confidence (all candidates) — Mahalanobis")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_dir / "novelty_vs_confidence_mahal.png", bbox_inches="tight")
        plt.close()

        # Top-K (by fused score) with tiers
        tier_colors = {
            "Tier 1": "green",
            "Tier 2": "orange",
            "Tier 3": "red",
            "Tier 4": "gray"
        }
        top_colors = df_top_fused["novelty_tier"].map(tier_colors)

        plt.figure(figsize=(7, 5))
        plt.scatter(
            df_top_fused["cosine_novelty"],
            df_top_fused["mahalanobis_novelty"],
            c=top_colors,
            edgecolor="k",
            alpha=0.85
        )
        for tier, col in tier_colors.items():
            plt.scatter([], [], color=col, label=tier)
        plt.xlabel("Cosine Novelty (Top-K Fused)")
        plt.ylabel("Mahalanobis Novelty (Top-K Fused)")
        plt.title("Top Candidates (Fused) with Tiers")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_dir / "topk_fused_cosine_vs_mahal.png", bbox_inches="tight")
        plt.close()

        # Agreement heatmaps
        plt.figure(figsize=(6, 5))
        plt.hexbin(
            df_cand["cosine_novelty"],
            df_cand["lof_novelty"],
            gridsize=40,
            cmap="plasma",
            mincnt=1,
        )
        plt.colorbar(label="Count")
        plt.xlabel("Cosine Novelty")
        plt.ylabel("LOF Novelty")
        plt.title("Agreement: LOF vs Cosine")
        plt.savefig(save_dir / "lof_vs_cosine_heatmap.png", bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.hexbin(
            df_cand["mahalanobis_novelty"],
            df_cand["lof_novelty"],
            gridsize=40,
            cmap="plasma",
            mincnt=1,
        )
        plt.colorbar(label="Count")
        plt.xlabel("Mahalanobis Novelty")
        plt.ylabel("LOF Novelty")
        plt.title("Agreement: LOF vs Mahalanobis")
        plt.savefig(save_dir / "lof_vs_mahal_heatmap.png", bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.hexbin(
            df_cand["cosine_novelty"],
            df_cand["mahalanobis_novelty"],
            gridsize=40,
            cmap="plasma",
            mincnt=1,
        )
        plt.colorbar(label="Count")
        plt.xlabel("Cosine Novelty")
        plt.ylabel("Mahalanobis Novelty")
        plt.title("Agreement: Cosine vs Mahalanobis")
        plt.savefig(save_dir / "cosine_vs_mahal_heatmap.png", bbox_inches="tight")
        plt.close()

        # Distributions
        plt.figure(figsize=(6, 4))
        plt.hist(df_cand["mean_probability"], bins=40, color="skyblue")
        plt.xlabel("Mean Probability")
        plt.ylabel("Count")
        plt.title("Distribution of Mean Probabilities")
        plt.grid(True)
        plt.savefig(save_dir / "mean_prob_distribution.png", bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.hist(df_cand["total_uncertainty"], bins=40, color="lightcoral")
        plt.xlabel("Total Uncertainty")
        plt.ylabel("Count")
        plt.title("Distribution of Uncertainty")
        plt.grid(True)
        plt.savefig(save_dir / "uncertainty_distribution.png", bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.hist(df_cand["lof_novelty"], bins=40, color="mediumseagreen")
        plt.xlabel("LOF Novelty Score")
        plt.ylabel("Count")
        plt.title("Distribution of LOF Novelty Scores")
        plt.grid(True)
        plt.savefig(save_dir / "lof_score_distribution.png", bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.hist(df_cand["mahalanobis_novelty"], bins=40, color="mediumpurple")
        plt.xlabel("Mahalanobis Novelty")
        plt.ylabel("Count")
        plt.title("Distribution of Mahalanobis Novelty")
        plt.grid(True)
        plt.savefig(save_dir / "mahal_score_distribution.png", bbox_inches="tight")
        plt.close()

    # Preserve your original return (probability-weighted ranking)
    return df_top_prob


def plot_tsne_umap_with_novel(
    embeddings: np.ndarray,
    known_pos_idx: np.ndarray,
    novel_pos_idx: np.ndarray,
    save_dir: str | Path,
    method: str = "tsne"
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    reducer = TSNE(n_components=2, perplexity=30, random_state=42) if method == "tsne" \
        else umap.UMAP(n_components=2, random_state=42)

    reduced = reducer.fit_transform(embeddings)
    xs, ys = reduced[:, 0], reduced[:, 1]

    plt.figure(figsize=(10, 7))

    # Background
    all_indices = np.arange(embeddings.shape[0])
    background_idx = np.setdiff1d(all_indices, np.union1d(known_pos_idx, novel_pos_idx))

    plt.scatter(xs[background_idx], ys[background_idx], c="lightgray", s=10, alpha=0.3, label="Other nodes", zorder=1)

    # Known Positives
    plt.scatter(xs[known_pos_idx], ys[known_pos_idx], c="blue", s=40, alpha=0.8,
                label=f"Known positives ({len(known_pos_idx)})", zorder=2)

    # Novel Positives
    plt.scatter(xs[novel_pos_idx], ys[novel_pos_idx], c="green", s=60, alpha=0.9,
                label=f"Novel positives ({len(novel_pos_idx)})", edgecolor="black", linewidth=0.6, zorder=3)

    plt.title(f"{method.upper()} - Known vs Novel Positives")
    plt.legend()
    plt.axis("off")

    out_path = save_dir / f"{method}_highlight_novel_positives_prob.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"📍 Saved {method.upper()} plot with novel positives → {out_path}")
