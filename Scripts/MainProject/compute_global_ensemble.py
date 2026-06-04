from Modules.embedding_loader import load_embeddings
from Modules.label_handler import load_labels
from Modules.rn_selector import kmeans_method, knn_method, gpu_method, ccrne_method
from Modules.rn_confidence_selector import select_confident_rns
from Modules.rn_visualizer import plot_rn_uncertainty_histograms
from Modules.rn_embedding_plotter import plot_rn_embedding_distribution
from Modules.pu_model_refinement import run_pu_refinement
from Modules.ensemble_uncertainty import compute_ensemble_uncertainty
from Modules.ensemble_uncertainty_plotter import (
    plot_uncertainty_histograms,
    plot_meanprob_vs_uncertainty,
    plot_uncertainty_boxplot
)
from Modules.analyze_final_rn_agreement import analyze_rn_overlap
from Modules.novel_positive_selector import *
from Modules.ensemble_uncertainty_plotter import plot_modelwise_epistemic_uncertainty
from Modules.ensemble_uncertainty_plotter import plot_combined_epistemic_uncertainty
from Modules.assign_final_novelty_tiers import (
    select_top_candidates_by_tier,
    plot_tsne_umap_with_tiers,
    plot_tsne_final_sets
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.special import softmax
from pathlib import Path
import torch
import torch, torch.nn as nn, torch.utils.data as tud
import numpy as np

# === Setup ===
method_name = 'DGI'
base_path = Path("/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG") / \
            f"VasculaidKG/CuratedData/Proteins/PAD/Results/{method_name.upper()}"

uncertainty_plot_dir = base_path / "UncertaintyPlots"
uncertainty_plot_dir.mkdir(parents=True, exist_ok=True)

avg_emb, std_emb = load_embeddings(method=method_name)

unsupervised_models = [
    'DGI', 'GAE', 'GRACE', 'SIMGRACE',
    'VGAE', 'MAE_DEGREE', 'MAE_ADAPTIVE', 'MAE_RANDOM'
]
all_embeddings = []
for model in unsupervised_models:
    avg_emb, std_emb = load_embeddings(method=model)  
    all_embeddings.append(avg_emb)

combined_emb = np.concatenate(all_embeddings, axis=1)
Xz = StandardScaler(with_mean=True, with_std=True).fit_transform(combined_emb)

# 2) PCA to a stable size for novelty metrics
n_novel = min(128, Xz.shape[1], max(1, Xz.shape[0]-1))  # guard if N < 129
pca = PCA(n_components=n_novel, svd_solver="randomized", random_state=42).fit(Xz)
emb_for_novelty = pca.transform(Xz)                     # [N, n_novel]

emb_for_tsne = emb_for_novelty[:, :min(50, emb_for_novelty.shape[1])]

print("Explained variance (novelty PCA):", pca.explained_variance_ratio_.sum())

label_csv_path = '/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/CuratedData/LargestComponents/Nodes_with_properties_FC.csv'
labels, known_mask, unknown_mask, known_indices, unknown_indices, labels_df = load_labels(label_csv_path)

# === PU Model Refinement ===
classifier_list = ['logistic_regression', 'svm', 'mc_dropout', 'bnn', 'random_forest']
pu_save_base = base_path
labels_np_base = labels.clone().numpy().flatten()

metadata_path = "/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/CuratedData/LargestComponents/Nodes_with_properties_FC.csv"
nodes_meta = pd.read_csv(metadata_path)

# === Global Ensemble Uncertainty ===
base_results_path = Path("/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Results")

classifiers = classifier_list
all_prediction_paths = []

nodes_meta = pd.read_csv(metadata_path)

for model in unsupervised_models:
    model_dir = base_results_path / model
    for clf in classifiers:
        pred_path = model_dir / clf.upper() / "FinalPredictions" / "final_predictions.csv"
        if pred_path.exists():
            print(f"✅ Found: {pred_path}")
            all_prediction_paths.append(pred_path)
        else:
            print(f"❌ Missing: {pred_path}")

if all_prediction_paths:
    df_all_ensemble = compute_ensemble_uncertainty(all_prediction_paths)

    # ✅ Rename for compatibility

    if 'node_index' in df_all_ensemble.columns and 'NodeIndex' in nodes_meta.columns:
        df_all_ensemble = df_all_ensemble.merge(
            nodes_meta[['NodeIndex', 'GeneSymbol']].rename(columns={'NodeIndex': 'node_index'}),
            on='node_index',
            how='left'
        )

    detailed_summary_path = base_results_path / "global_ensemble_uncertainty_summary_detailed.csv"
    df_all_ensemble.to_csv(detailed_summary_path, index=False)
    print(f"✅ Saved detailed global ensemble uncertainty summary to: {detailed_summary_path}")

    summary_path = base_results_path / "global_ensemble_uncertainty_summary.csv"
    df_all_ensemble[['node_index', 'mean_probability', 'epistemic_uncertainty', 'aleatoric_uncertainty', 'total_uncertainty', 'predicted_label']].to_csv(summary_path, index=False)

    plot_dir = base_results_path / "EnsembleUncertaintyPlots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_uncertainty_histograms(df_all_ensemble, save_path=plot_dir / "histogram.png", title_prefix="Global")
    plot_uncertainty_boxplot(df_all_ensemble, save_path=plot_dir / "boxplot.png", title_prefix="Global")
    plot_modelwise_epistemic_uncertainty(df_all_ensemble, save_dir=plot_dir, title_prefix="Global")
    plot_combined_epistemic_uncertainty(df_all_ensemble, save_path=plot_dir / "combined_epistemic_uncertainty.png", title_prefix="Global")

    print(f"📊 Saved ensemble uncertainty plots in: {plot_dir}")

    # === Final Novel Positive Selection ===
    novel_pos_dir = base_results_path / "Novel_Positive_Analysis"
    novel_pos_dir.mkdir(parents=True, exist_ok=True)

    df_top_candidates = compute_novel_positive_candidates(
        ensemble_csv_path=detailed_summary_path,
        known_positive_indices=known_indices,
        embeddings=combined_emb,
        save_dir=novel_pos_dir, 
        min_mean_prob=0.75,
        max_uncertainty=0.40,
        top_k=50,
        lof_k=20,
        generate_plots=True     
    )
    plot_meanprob_vs_uncertainty(df_all=df_all_ensemble, df_top=df_top_candidates, save_path=plot_dir / "meanprob_vs_uncertainty_highlighted.png", title_prefix="Global")
    plot_uncertainty_histograms(df_all_ensemble, save_path=plot_dir / "histogram.png", title_prefix="Global")
    plot_uncertainty_boxplot(df_all_ensemble, save_path=plot_dir / "boxplot.png", title_prefix="Global")
    plot_modelwise_epistemic_uncertainty(df_all_ensemble, save_dir=plot_dir, title_prefix="Global")
    plot_combined_epistemic_uncertainty(df_all_ensemble, save_path=plot_dir / "combined_epistemic_uncertainty.png", title_prefix="Global")

    print("Top novel candidates:", df_top_candidates.shape[0])
    print("Node indices:", df_top_candidates['node_index'].values)

    if not df_top_candidates.empty:
        plot_tsne_umap_with_novel(
            embeddings=combined_emb,
            known_pos_idx=known_indices,
            novel_pos_idx=df_top_candidates["node_index"].to_numpy(),
            save_dir=novel_pos_dir,
            method="tsne"  # or "umap"
        )

    # === Load all candidates from previous output
    all_candidates_path = novel_pos_dir / "all_positive_candidates.csv"
    df_all_candidates = pd.read_csv(all_candidates_path)

    # === Assign final tiers and select capped top 50
    df_top_50, df_all_with_tiers = select_top_candidates_by_tier(df_all_candidates)

    # === Save both full and top-tiered candidate tables
    df_top_50.to_csv(novel_pos_dir / "top_50_tiered_candidates.csv", index=False)
    df_all_with_tiers.to_csv(novel_pos_dir / "all_candidates_with_tiers.csv", index=False)

    print(f"✅ Saved top 50 tiered candidates to: {novel_pos_dir / 'top_50_tiered_candidates.csv'}")
    print(f"📋 Saved full candidate table with tiers to: {novel_pos_dir / 'all_candidates_with_tiers.csv'}")

    # === Plot with tiers annotated
    plot_tsne_umap_with_tiers(
        embeddings=combined_emb,
        known_pos_idx=known_indices,
        tiered_df=df_top_50,
        save_dir=novel_pos_dir,
        method="tsne"
    )

    final_sel_dir = Path("/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Results/Novel_Positive_Analysis/Novel_candidates_ablation/FinalSelections")
    plot_tsne_final_sets(
        embeddings=combined_emb,          # use your concatenated embeddings for nicer geometry
        known_pos_idx=known_indices,      # from load_labels(...)
        results_dir=final_sel_dir,
        save_path=final_sel_dir / "tsne_final_sets.png",
        perplexity=30,
        random_state=42,
    )

    print(f"✅ Final novel positives saved in: {novel_pos_dir}")
else:
    print("❌ No prediction files found across all models.")

# === Final RN Agreement Analysis ===
analyze_rn_overlap()
