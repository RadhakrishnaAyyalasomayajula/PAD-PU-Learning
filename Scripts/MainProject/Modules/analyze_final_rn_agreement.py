import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import combinations

def load_final_rns(results_base_path, model_name, classifiers):
    rn_dict = {}
    for clf in classifiers:
        path = Path(results_base_path) / model_name / clf.upper() / "RefinedLabels" / "final_labels_with_index.csv"
        if path.exists():
            df = pd.read_csv(path)
            rns = df[df['final_label'] == 0]['node_index'].tolist()
            rn_dict[clf] = set(rns)
        else:
            print(f"⚠️ Missing file for {clf}: {path}")
    return rn_dict

def compute_jaccard_matrix(rn_dict):
    classifiers = list(rn_dict.keys())
    n = len(classifiers)
    overlap_matrix = np.zeros((n, n), dtype=int)
    jaccard_matrix = np.zeros((n, n), dtype=float)

    for i, j in combinations(range(n), 2):
        set_i = rn_dict[classifiers[i]]
        set_j = rn_dict[classifiers[j]]
        intersection = len(set_i & set_j)
        union = len(set_i | set_j)
        jaccard = intersection / union if union > 0 else 0
        overlap_matrix[i, j] = overlap_matrix[j, i] = intersection
        jaccard_matrix[i, j] = jaccard_matrix[j, i] = jaccard

    for k in range(n):
        overlap_matrix[k, k] = len(rn_dict[classifiers[k]])
        jaccard_matrix[k, k] = 1.0

    return classifiers, overlap_matrix, jaccard_matrix

def plot_heatmap(matrix, labels, title, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt=".2f", cmap="viridis")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_rn_overlap():
    results_base = "/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Results"
    model_name = "DGI"
    classifiers = ['logistic_regression', 'svm', 'mc_dropout', 'bnn', 'random_forest']
    output_dir = Path(results_base) / model_name
    os.makedirs(output_dir, exist_ok=True)

    rn_dict = load_final_rns(results_base, model_name, classifiers)
    labels, overlap_matrix, jaccard_matrix = compute_jaccard_matrix(rn_dict)

    pd.DataFrame(overlap_matrix, index=labels, columns=labels).to_csv(output_dir / "RN_overlap_matrix.csv")
    pd.DataFrame(jaccard_matrix, index=labels, columns=labels).to_csv(output_dir / "RN_jaccard_similarity_matrix.csv")

    plot_heatmap(overlap_matrix, labels, "Final RN Overlap (Counts)", output_dir / "RN_overlap_heatmap.png")
    plot_heatmap(jaccard_matrix, labels, "Final RN Jaccard Similarity", output_dir / "RN_jaccard_similarity_heatmap.png")

    print(f"✅ Saved RN overlap & similarity results to: {output_dir}")
