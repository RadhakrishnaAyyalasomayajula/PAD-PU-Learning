import matplotlib.pyplot as plt
import pandas as pd

def plot_uncertainty_histograms(df, save_path=None, title_prefix=""):
    plt.figure(figsize=(10, 6))
    plt.hist(df['aleatoric_uncertainty'], bins=50, alpha=0.6, label='Aleatoric')
    plt.hist(df['epistemic_uncertainty'], bins=50, alpha=0.6, label='Epistemic')
    plt.hist(df['total_uncertainty'], bins=50, alpha=0.6, label='Total')
    plt.xlabel("Uncertainty")
    plt.ylabel("Frequency")
    plt.title(f"{title_prefix} Ensemble Uncertainty Histogram")
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

import matplotlib.pyplot as plt

def plot_meanprob_vs_uncertainty(
    df_all: pd.DataFrame,
    df_top: pd.DataFrame | None = None,
    save_path=None,
    title_prefix=""
):
    x_col = 'mean_probability'
    if x_col not in df_all.columns:
        if 'prob_positive_mean' in df_all.columns:
            x_col = 'prob_positive_mean'
        else:
            raise KeyError("Neither 'mean_probability' nor 'prob_positive_mean' found in DataFrame.")

    plt.figure(figsize=(8, 6))

    # Aleatoric in light gray
    plt.scatter(
        df_all[x_col],
        df_all['aleatoric_uncertainty'],
        alpha=0.3,
        label='Aleatoric Uncertainty',
        color='green'
    )

    # Epistemic in dark gray
    plt.scatter(
        df_all[x_col],
        df_all['epistemic_uncertainty'],
        alpha=0.4,
        label='Epistemic Uncertainty',
        color='gray'
    )

    if df_top is not None:
        # Aleatoric layer
        plt.scatter(
            df_top[x_col],
            df_top['aleatoric_uncertainty'],
            color='blue',
            edgecolor='black',
            label='Top Novel Positives (Aleatoric)',
            s=40,
            zorder=5
        )

        # Epistemic layer
        plt.scatter(
            df_top[x_col],
            df_top['epistemic_uncertainty'],
            color='green',
            edgecolor='black',
            label='Top Novel Positives (Epistemic)',
            s=40,
            marker='D',
            zorder=6
        )

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Uncertainty")
    plt.title(f"{title_prefix} Mean Prob vs. Uncertainty")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_uncertainty_boxplot(df, save_path=None, title_prefix=""):
    data = [
        df['aleatoric_uncertainty'],
        df['epistemic_uncertainty'],
        df['total_uncertainty']
    ]
    labels = ['Aleatoric', 'Epistemic', 'Total']
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels, patch_artist=True)
    plt.ylabel("Uncertainty")
    plt.title(f"{title_prefix} Uncertainty Boxplot")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_modelwise_epistemic_uncertainty(df, save_dir, title_prefix=""):
    if 'epistemic_mc_avg' in df.columns:
        plt.figure(figsize=(7, 4))
        plt.hist(df['epistemic_mc_avg'], bins=40, color='skyblue')
        plt.title(f"{title_prefix} MC Dropout Epistemic Uncertainty")
        plt.xlabel("Epistemic Uncertainty")
        plt.ylabel("Count")
        plt.grid(True)
        plt.savefig(save_dir / "mc_epistemic_uncertainty.png", bbox_inches="tight")
        plt.close()

    if 'epistemic_bnn_avg' in df.columns:
        plt.figure(figsize=(7, 4))
        plt.hist(df['epistemic_bnn_avg'], bins=40, color='lightgreen')
        plt.title(f"{title_prefix} BNN Epistemic Uncertainty")
        plt.xlabel("Epistemic Uncertainty")
        plt.ylabel("Count")
        plt.grid(True)
        plt.savefig(save_dir / "bnn_epistemic_uncertainty.png", bbox_inches="tight")
        plt.close()

def plot_combined_epistemic_uncertainty(df, save_path=None, title_prefix=""):
    plt.figure(figsize=(10, 6))

    if 'epistemic_uncertainty' in df.columns:
        plt.hist(df['epistemic_uncertainty'], bins=50, alpha=0.5, label='Ensemble', density=True)

    if 'epistemic_mc_avg' in df.columns:
        plt.hist(df['epistemic_mc_avg'], bins=50, alpha=0.5, label='MC Dropout', density=True)

    if 'epistemic_bnn_avg' in df.columns:
        plt.hist(df['epistemic_bnn_avg'], bins=50, alpha=0.5, label='BNN', density=True)

    plt.xlabel("Epistemic Uncertainty")
    plt.ylabel("Density")
    plt.title(f"{title_prefix} Comparison of Epistemic Uncertainties")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()