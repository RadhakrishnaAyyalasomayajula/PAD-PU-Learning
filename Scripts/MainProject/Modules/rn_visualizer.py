import os
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

def plot_rn_uncertainty_histograms(rn_scores, save_dir=None, prefix="rn_uncertainty"):
    """
    Plot and optionally save histograms of RN uncertainty metrics.

    Args:
        rn_scores (dict): Output from select_confident_rns (contains score dicts).
        save_dir (str, optional): Directory to save plots. If None, plots aren't saved.
        prefix (str): Filename prefix if saving.
    """
    # Use agreement_raw for proper integer counts
    agreement_vals = list(rn_scores['agreement_raw'].values())
    entropy_vals   = list(rn_scores['entropy'].values())
    distance_vals  = list(rn_scores['distance'].values())
    combined_vals  = list(rn_scores['combined'].values())

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()

    # --- Agreement (bar plot) ---
    agreement_counter = Counter(agreement_vals)
    agreement_levels = sorted(agreement_counter.keys())
    agreement_counts = [agreement_counter[k] for k in agreement_levels]

    axs[0].bar(agreement_levels, agreement_counts, color='skyblue', edgecolor='k')
    axs[0].set_title("Agreement Score Distribution")
    axs[0].set_xlabel("Number of Methods Agreeing")
    axs[0].set_ylabel("RN Candidates")
    axs[0].set_xticks(agreement_levels)

    # --- Entropy ---
    axs[1].hist(entropy_vals, bins=30, color='salmon', edgecolor='k')
    axs[1].set_title("Entropy-Based Uncertainty")
    axs[1].set_xlabel("Shannon Entropy")
    axs[1].set_ylabel("Frequency")

    # --- Distance ---
    axs[2].hist(distance_vals, bins=30, color='lightgreen', edgecolor='k')
    axs[2].set_title("Distance-Based Uncertainty")
    axs[2].set_xlabel("Avg Distance to Positives")
    axs[2].set_ylabel("Frequency")

    # --- Combined ---
    axs[3].hist(combined_vals, bins=30, color='orchid', edgecolor='k')
    axs[3].set_title("Combined RN Confidence Score")
    axs[3].set_xlabel("Confidence Score (0–1)")
    axs[3].set_ylabel("RN Candidates")

    plt.tight_layout()
    plt.show()

    # --- Save plots individually ---
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # Agreement
        plt.figure(figsize=(6, 4))
        plt.bar(agreement_levels, agreement_counts, color='skyblue', edgecolor='k')
        plt.xlabel("Number of Methods Agreeing")
        plt.ylabel("RN Candidates")
        plt.title("Agreement Score Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_agreement.png"))
        plt.close()

        # Others
        for name, vals, bins, color, xlabel in [
            ("entropy", entropy_vals, 30, "salmon", "Shannon Entropy"),
            ("distance", distance_vals, 30, "lightgreen", "Avg Distance to Positives"),
            ("combined", combined_vals, 30, "orchid", "Confidence Score (0–1)")
        ]:
            plt.figure(figsize=(6, 4))
            plt.hist(vals, bins=bins, color=color, edgecolor='k')
            plt.xlabel(xlabel)
            plt.ylabel("Frequency")
            plt.title(f"{name.capitalize()} Histogram")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{prefix}_{name}.png"))
            plt.close()