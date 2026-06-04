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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.special import softmax
from pathlib import Path
import torch

# === Setup ===
method_name = 'DGI'
base_path = Path("/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG") / \
            f"VasculaidKG/CuratedData/Proteins/PAD/Results/{method_name.upper()}"

uncertainty_plot_dir = base_path / "UncertaintyPlots"
uncertainty_plot_dir.mkdir(parents=True, exist_ok=True)

avg_emb, std_emb = load_embeddings(method=method_name)

label_csv_path = '/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/CuratedData/LargestComponents/Nodes_with_properties_FC.csv'
labels, known_mask, unknown_mask, known_indices, unknown_indices, labels_df = load_labels(label_csv_path)

rn_ratio = 1.0
max_init_rn_ratio = 2.0
num_known_positives = (labels == 1).sum().item()
max_final_rns = int(max_init_rn_ratio * num_known_positives)

# === RN Selection ===
rn_kmeans = kmeans_method(avg_emb, labels, rn_ratio=rn_ratio)
rn_knn = knn_method(avg_emb, labels, rn_ratio=rn_ratio)
rn_gpu = gpu_method(avg_emb, labels, rn_ratio=rn_ratio)
rn_ccrne = ccrne_method(avg_emb, labels, rn_ratio=rn_ratio)
rn_sets = [rn_kmeans, rn_knn, rn_gpu, rn_ccrne]

embedding_probs = softmax(avg_emb, axis=1)

final_rns, rn_scores, all_rn_candidates, combined = select_confident_rns(
    rn_sets=rn_sets,
    embeddings=avg_emb,
    labels=labels,
    embedding_probs=embedding_probs,
    top_k_ratio=1.0,
    k_distance=5,
    weight_scheme="custom",
    max_final_rns=max_final_rns,
    export_base=base_path
)

plot_rn_uncertainty_histograms(rn_scores, save_dir=str(uncertainty_plot_dir), prefix=method_name.lower())

# Backup and update labels
backup_path = base_path / "backup_labels_before_rn_update.npy"
np.save(backup_path, labels.numpy())

labels[final_rns] = 0
np.save(base_path / "updated_labels_with_rns.npy", labels.numpy())

print("Label breakdown before initial TSNE:", np.unique(labels.numpy(), return_counts=True))
print("Final RN count:", len(final_rns))

plot_rn_embedding_distribution(
    embeddings=avg_emb,
    labels_tensor=labels,
    rn_indices=final_rns,
    method="TSNE",
    save_path=str(base_path / "embedding_TSNE.png")
)

# RN Method Details
rn_dict = {"kmeans": rn_kmeans, "knn": rn_knn, "gpu": rn_gpu, "ccrne": rn_ccrne}
score_lookup = dict(zip(all_rn_candidates, combined))
for key in rn_dict:
    rn_dict[key] = sorted(rn_dict[key], key=lambda x: -score_lookup.get(x, 0))

max_len = max(len(rns) for rns in rn_dict.values())
padded_data = {
    k: np.pad(np.array(v, dtype=float), (0, max_len - len(v)), constant_values=np.nan)
    for k, v in rn_dict.items()
}
pd.DataFrame(padded_data).to_csv(base_path / "rn_candidates_by_method.csv", index=False)

# === PU Model Refinement ===
classifier_list = ['logistic_regression', 'svm', 'mc_dropout', 'bnn', 'random_forest']
pu_save_base = base_path
labels_np_base = labels.clone().numpy().flatten()

metadata_path = "/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/CuratedData/LargestComponents/Nodes_with_properties_FC.csv"
nodes_meta = pd.read_csv(metadata_path)

for clf_name in classifier_list:
    print(f"\n>>> Starting PU refinement using classifier: {clf_name.upper()}")
    labels_np = labels_np_base.copy()

    run_pu_refinement(
        embeddings=avg_emb,
        labels=labels_np,
        classifier_name=clf_name,
        save_dir=pu_save_base / clf_name.upper(),
        max_iter=10,
        min_new_rns=5,
        uncertainty_threshold=0.05,
        prob_threshold=0.2
    )

    refined_label_path = pu_save_base / clf_name.upper() / "RefinedLabels" / "refined_labels.npy"
    if refined_label_path.exists():
        refined_labels = np.load(refined_label_path)
        rn_indices = np.where(refined_labels == 0)[0]

        rn_meta = nodes_meta.loc[rn_indices, ['NodeIndex', 'GeneSymbol']].copy()
        rn_meta['RN_Label'] = 0
        rn_meta['Classifier'] = clf_name

        rn_meta.to_csv(pu_save_base / clf_name.upper() / "FinalRNs_with_GeneSymbols.csv", index=False)
        print(f"✅ Saved final RNs for {clf_name.upper()}")

        plot_rn_embedding_distribution(
            embeddings=avg_emb,
            labels_tensor=torch.tensor(refined_labels),
            rn_indices=rn_indices,
            method="TSNE",
            save_path=str(pu_save_base / clf_name.upper() / "tsne_with_rns_and_positives.png")
        )

# === Global Ensemble Uncertainty ===
base_results_path = Path("/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Results")
unsupervised_models = [method_name]
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
        embeddings=avg_emb,
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
            embeddings=avg_emb,
            known_pos_idx=known_indices,
            novel_pos_idx=df_top_candidates["node_index"].to_numpy(),
            save_dir=novel_pos_dir,
            method="tsne"  # or "umap"
        )

    print(f"✅ Final novel positives saved in: {novel_pos_dir}")
else:
    print("❌ No prediction files found across all models.")

# === Final RN Agreement Analysis ===
analyze_rn_overlap()
