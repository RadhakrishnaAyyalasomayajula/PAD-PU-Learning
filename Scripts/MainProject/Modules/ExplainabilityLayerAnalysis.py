import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

# ======================================================================
# CONFIG / PATHS
# ======================================================================
BASE_DIR = Path("/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD")
NODES_FILE = BASE_DIR / "CuratedData/LargestComponents/Nodes_with_properties_FC.csv"
EDGES_FILE = BASE_DIR / "CuratedData/LargestComponents/Edges_with_nodeproperties_FC.csv"
CANDIDATES_FILE = BASE_DIR / "Results/ExplainabilityAnalysis/Candidates/FinalCandidates_Explainability.csv"
EXPL_DIR = BASE_DIR / "Results/ExplainabilityAnalysis/Explanations/"
PLOTS_BASE = EXPL_DIR / "Plots"
EXPL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_BASE.mkdir(parents=True, exist_ok=True)

# Subfolders per group
GROUPS = {
    "prob": {
        "name": "Probability-heavy",
        "plot_dir": PLOTS_BASE / "prob",
        "save_dir": EXPL_DIR / "Group_prob",
        "candidate_slice": slice(0, 50)       # first 50 = probability heavy
    },
    "novelty": {
        "name": "Novelty-heavy",
        "plot_dir": PLOTS_BASE / "novelty",
        "save_dir": EXPL_DIR / "Group_novelty",
        "candidate_slice": slice(50, 100)     # last 50 = novelty heavy
    }
}
for g in GROUPS.values():
    g["plot_dir"].mkdir(parents=True, exist_ok=True)
    g["save_dir"].mkdir(parents=True, exist_ok=True)

# ======================================================================
# LOAD METADATA
# ======================================================================
nodes_df = pd.read_csv(NODES_FILE)
edges_df = pd.read_csv(EDGES_FILE)
candidates_df = pd.read_csv(CANDIDATES_FILE)

# Build full graph
G = nx.from_pandas_edgelist(edges_df, "Index1", "Index2", create_using=nx.Graph())

# Known PAD labels (column PAD = 1)
KNOWN_PAD = set(nodes_df.index[nodes_df["PAD"] == 1].tolist())

# The initial 100 candidates (predicted positives) for consistent annotation
INITIAL_CANDIDATE_SET = set(candidates_df["node_index"].astype(int).tolist())

# ======================================================================
# LOAD EMBEDDINGS TO CONSTRUCT FEATURE NAMES
# ======================================================================
aaseq = np.load(BASE_DIR / "Scripts/aaseq_embeddings.npy").squeeze(1)
ntseq = np.load(BASE_DIR / "Scripts/ntseq_embeddings.npy").squeeze(1)
func  = np.load(BASE_DIR / "Scripts/function_embeddings.npy").squeeze(1)

aaseq_dim, ntseq_dim, func_dim = aaseq.shape[1], ntseq.shape[1], func.shape[1]
num_numeric = 3
total_features = aaseq_dim + ntseq_dim + func_dim + num_numeric

feature_names = (
    [f"AASeq_emb_{i}"  for i in range(aaseq_dim)] +
    [f"NTSeq_emb_{i}"  for i in range(ntseq_dim)] +
    [f"Func_emb_{i}"   for i in range(func_dim)] +
    ['MolWeight', 'AASeqLen', 'NTSeqLen']
)

# feature group slices
SL_AASEQ = slice(0, aaseq_dim)
SL_NTSEQ = slice(aaseq_dim, aaseq_dim + ntseq_dim)
SL_FUNC  = slice(aaseq_dim + ntseq_dim, aaseq_dim + ntseq_dim + func_dim)
IDX_MW, IDX_AAL, IDX_NTL = total_features - 3, total_features - 2, total_features - 1

# ======================================================================
# UTILITIES
# ======================================================================
def list_npz_files():
    return list(EXPL_DIR.glob("*.npz"))

def load_masks_for_node(node_idx, npz_files):
    masks_node, masks_edge = [], []
    needle = f"node{node_idx}"
    for f in npz_files:
        if needle in f.stem:
            data = np.load(f)
            masks_edge.append(data["edge_mask"])
            masks_node.append(data["node_mask"])
    return masks_edge, masks_node

def compute_feature_quality(node_mask, topk=10):
    top_idx = np.argsort(node_mask)[-topk:]
    embed_count = np.sum(top_idx < total_features - 3)  # last 3 numeric
    return float(embed_count) / topk if topk > 0 else 0.0

def pad_neighbor_support(node_idx):
    neigh = list(G.neighbors(node_idx))
    if not neigh:
        return 0
    return int(nodes_df.loc[neigh, "PAD"].sum())

def stability_across_models(node_idx, npz_files):
    _, node_masks = load_masks_for_node(node_idx, npz_files)
    clean = []
    for nm in node_masks:
        nm = np.array(nm)
        if nm.ndim == 2 and nm.shape[0] == len(nodes_df):
            clean.append(nm[node_idx, :])
        elif nm.ndim == 1 and nm.shape[0] == total_features:
            clean.append(nm)
    if len(clean) < 2:
        return 1.0
    arr = np.stack(clean, axis=0)
    return float(arr.var(axis=0).mean())

def unfaithfulness_score(node_idx, node_mask, k=10):
    # proxy: how much of node_mask mass is in top-k features, scaled by mean_probability
    if node_mask is None or node_mask.sum() <= 0:
        return 0.0
    top_idx = np.argsort(node_mask)[-k:]
    removed_fraction = node_mask[top_idx].sum() / (node_mask.sum() + 1e-9)
    baseline_prob = float(candidates_df.loc[candidates_df["node_index"] == node_idx, "mean_probability"].values[0])
    perturbed_prob = baseline_prob * (1 - removed_fraction)
    return (baseline_prob - perturbed_prob) / (baseline_prob + 1e-9)

def tag_of_node(i: int) -> str:
    if i in KNOWN_PAD:
        return "KNOWN"
    elif i in INITIAL_CANDIDATE_SET:
        return "CANDIDATE"
    else:
        return "OTHER"

def color_of_tag(tag: str) -> str:
    if tag == "KNOWN":
        return "#e74c3c"   # red
    elif tag == "CANDIDATE":
        return "#f39c12"   # orange
    return "#3498db"       # blue

def label_of_node(i: int) -> str:
    # only gene symbol (no [KNOWN]/[CANDIDATE] tags in text)
    return str(nodes_df.loc[i, "GeneSymbol"])

def label_of_edge(e_idx: int) -> str:
    # Gene1--Gene2 (no tags)
    u = int(edges_df.loc[e_idx, "Index1"])
    v = int(edges_df.loc[e_idx, "Index2"])
    su, sv = nodes_df.loc[u, "GeneSymbol"], nodes_df.loc[v, "GeneSymbol"]
    return f"{su}--{sv}"

def color_of_edge(e_idx: int) -> str:
    # color by presence of KNOWN or CANDIDATE endpoints
    u = int(edges_df.loc[e_idx, "Index1"])
    v = int(edges_df.loc[e_idx, "Index2"])
    tu, tv = tag_of_node(u), tag_of_node(v)
    if tu == "KNOWN" or tv == "KNOWN":
        return color_of_tag("KNOWN")
    if tu == "CANDIDATE" or tv == "CANDIDATE":
        return color_of_tag("CANDIDATE")
    return color_of_tag("OTHER")

def plot_top_bar(series, title, fname, label_fn=None, color_fn=None, topn=10, exclude_indices=None):
    s = pd.Series(series)
    if exclude_indices is not None:
        s.loc[list(exclude_indices)] = -np.inf
    top_idx = s.sort_values(ascending=False).head(topn).index.tolist()
    top_vals = s.loc[top_idx].values
    labels = [label_fn(i) if label_fn else str(i) for i in top_idx]
    colors = [color_fn(i) if color_fn else None for i in top_idx]
    plt.figure(figsize=(9, 6))
    plt.barh(range(len(top_vals)), top_vals, color=colors if color_fn else None)
    plt.gca().invert_yaxis()
    plt.yticks(range(len(top_vals)), labels, fontsize=9)
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()

def draw_subgraph(center_node, title, fname):
    neigh = list(G.neighbors(center_node))
    sub_nodes = [center_node] + neigh
    subG = G.subgraph(sub_nodes)
    node_colors = [color_of_tag(tag_of_node(n)) for n in subG.nodes()]
    node_edges = ["#000000" if n == center_node else "#333333" for n in subG.nodes()]
    plt.figure(figsize=(6.5, 6.5))
    pos = nx.spring_layout(subG, seed=17)
    nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=720, linewidths=2, edgecolors=node_edges)
    nx.draw_networkx_edges(subG, pos, alpha=0.6)
    nx.draw_networkx_labels(subG, pos, labels={n: label_of_node(n) for n in subG.nodes()}, font_size=8, font_weight="bold")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()

# ======================================================================
# PER-GROUP PIPELINE
# ======================================================================
def run_group_pipeline(group_key):
    cfg = GROUPS[group_key]
    group_name = cfg["name"]
    plot_dir, save_dir = cfg["plot_dir"], cfg["save_dir"]
    print(f"\n=== Running group: {group_name} ===")

    group_candidates_df = candidates_df.iloc[cfg["candidate_slice"]].copy()
    group_candidate_indices = group_candidates_df["node_index"].astype(int).tolist()
    npz_files_all = list_npz_files()
    if len(npz_files_all) == 0:
        raise RuntimeError("No .npz explanation files found")

    def aggregate_for_candidates(candidate_indices):
        edge_sum, node_sum = None, None
        node_context_sum = np.zeros(len(nodes_df), dtype=np.float64)
        count = 0
        records = []

        for node_idx in candidate_indices:
            edge_masks, node_masks = load_masks_for_node(node_idx, npz_files_all)
            if len(edge_masks) == 0 or len(node_masks) == 0:
                continue

            edge_avg = np.mean(np.stack(edge_masks, axis=0), axis=0).astype(np.float32)

            clean_masks = []
            for nm in node_masks:
                nm = np.array(nm)
                if nm.ndim == 2 and nm.shape[0] == len(nodes_df):
                    clean_masks.append(nm[node_idx, :])
                elif nm.ndim == 1 and nm.shape[0] == total_features:
                    clean_masks.append(nm)
            node_avg = np.mean(np.stack(clean_masks, axis=0), axis=0).astype(np.float32)

            if edge_sum is None:
                edge_sum = np.zeros_like(edge_avg, dtype=np.float64)
            if node_sum is None:
                node_sum = np.zeros((len(nodes_df), node_avg.shape[0]), dtype=np.float64)

            edge_sum += edge_avg
            node_sum[node_idx, :] += node_avg
            count += 1

            # distribute edge importance to incident nodes for "context"
            for e_idx, score in enumerate(edge_avg):
                u = int(edges_df.loc[e_idx, "Index1"])
                v = int(edges_df.loc[e_idx, "Index2"])
                node_context_sum[u] += score
                node_context_sum[v] += score

            # per-candidate explanation metrics
            feat_quality = compute_feature_quality(node_avg)
            pad_support = pad_neighbor_support(node_idx)
            stab = stability_across_models(node_idx, npz_files_all)
            unfaith = unfaithfulness_score(node_idx, node_avg)

            expl_score = 0.35 * feat_quality + 0.25 * (pad_support > 0) + 0.25 * (1 - stab) + 0.15 * (1 - unfaith)

            records.append({
                "node_index": node_idx,
                "GeneSymbol": nodes_df.loc[node_idx, "GeneSymbol"],
                "feature_quality": feat_quality,
                "pad_support": pad_support,
                "stability": stab,
                "unfaithfulness": unfaith,
                "explanation_score": expl_score
            })

        if count == 0:
            raise RuntimeError("No explanations aggregated for this set.")

        edge_consensus = edge_sum / count
        node_consensus = node_sum / count
        feature_consensus = node_consensus.mean(axis=0)
        node_context_consensus = node_context_sum / count

        return edge_consensus, feature_consensus, node_context_consensus, pd.DataFrame(records)

    # --- RAW (all 50) ---
    edge_cons_raw, feat_cons_raw, ctx_cons_raw, df_rank_raw = aggregate_for_candidates(group_candidate_indices)
    df_rank_raw.sort_values("explanation_score", ascending=False, inplace=True)
    df_rank_raw.to_csv(save_dir / "candidates_explainability_ranked_raw.csv", index=False)

    # --- FILTERED (top-25 by explanation score) ---
    top25 = df_rank_raw.head(25)["node_index"].tolist()
    edge_cons_filt, feat_cons_filt, ctx_cons_filt, df_rank_filt = aggregate_for_candidates(top25)
    df_rank_filt.sort_values("explanation_score", ascending=False, inplace=True)
    df_rank_filt.to_csv(save_dir / "candidates_explainability_ranked_filtered.csv", index=False)

    # -------------------------
    # Save consensus CSVs
    # -------------------------
    pd.DataFrame({"edge_importance": edge_cons_raw}).to_csv(save_dir / "edge_importance_consensus_raw.csv", index=False)
    pd.DataFrame({"edge_importance": edge_cons_filt}).to_csv(save_dir / "edge_importance_consensus_filtered.csv", index=False)

    pd.Series(feat_cons_raw, index=feature_names, name="importance").to_csv(save_dir / "feature_importance_consensus_raw.csv")
    pd.Series(feat_cons_filt, index=feature_names, name="importance").to_csv(save_dir / "feature_importance_consensus_filtered.csv")

    pd.Series(ctx_cons_raw, index=nodes_df.index, name="context_importance").to_csv(save_dir / "node_context_importance_raw.csv")
    pd.Series(ctx_cons_filt, index=nodes_df.index, name="context_importance").to_csv(save_dir / "node_context_importance_filtered.csv")

    # Grouped features (SUM, not mean)
    def grouped_from_feature_consensus(fc):
        return {
            "AASeq_embeddings": float(fc[SL_AASEQ].sum()),
            "NTSeq_embeddings": float(fc[SL_NTSEQ].sum()),
            "Function_embeddings": float(fc[SL_FUNC].sum()),
            "MolWeight": float(fc[IDX_MW]),
            "AASeqLen": float(fc[IDX_AAL]),
            "NTSeqLen": float(fc[IDX_NTL]),
        }

    pd.Series(grouped_from_feature_consensus(feat_cons_raw), name="importance").to_csv(save_dir / "node_importance_grouped_raw.csv")
    pd.Series(grouped_from_feature_consensus(feat_cons_filt), name="importance").to_csv(save_dir / "node_importance_grouped_filtered.csv")

    # -------------------------
    # PLOTS
    # -------------------------
    plot_top_bar(grouped_from_feature_consensus(feat_cons_raw),
                 f"{group_name}: Top Feature Groups (raw)",
                 plot_dir / "top_feature_groups_raw.png",
                 topn=6)

    plot_top_bar(grouped_from_feature_consensus(feat_cons_filt),
                 f"{group_name}: Top Feature Groups (filtered)",
                 plot_dir / "top_feature_groups_filtered.png",
                 topn=6)

    plot_top_bar(pd.Series(edge_cons_raw),
                 f"{group_name}: Top-10 Edges (raw)",
                 plot_dir / "top10_edges_raw.png",
                 label_fn=label_of_edge,
                 color_fn=color_of_edge,
                 topn=10)

    plot_top_bar(pd.Series(edge_cons_filt),
                 f"{group_name}: Top-10 Edges (filtered)",
                 plot_dir / "top10_edges_filtered.png",
                 label_fn=label_of_edge,
                 color_fn=color_of_edge,
                 topn=10)

    plot_top_bar(pd.Series(ctx_cons_raw),
                 f"{group_name}: Top-10 Context Nodes (raw, excl. candidates)",
                 plot_dir / "top10_context_nodes_raw.png",
                 label_fn=label_of_node,
                 color_fn=lambda i: color_of_tag(tag_of_node(i)),
                 topn=10,
                 exclude_indices=INITIAL_CANDIDATE_SET)

    plot_top_bar(pd.Series(ctx_cons_filt),
                 f"{group_name}: Top-10 Context Nodes (filtered, excl. candidates)",
                 plot_dir / "top10_context_nodes_filtered.png",
                 label_fn=label_of_node,
                 color_fn=lambda i: color_of_tag(tag_of_node(i)),
                 topn=10,
                 exclude_indices=INITIAL_CANDIDATE_SET)

    # -------------------------
    # SUBGRAPHS
    # -------------------------
    # Top-5 candidates (raw)
    top5_cands_raw = df_rank_raw["node_index"].head(5).tolist()
    for rank, n in enumerate(top5_cands_raw, start=1):
        draw_subgraph(
            n,
            f"{group_name}: Candidate #{rank} (raw) — {label_of_node(n)}",
            plot_dir / f"subgraph_candidate_raw_{rank}_{nodes_df.loc[n,'GeneSymbol']}.png"
        )

    # Top-5 candidates (filtered)
    top5_cands_filt = df_rank_filt["node_index"].head(5).tolist()
    for rank, n in enumerate(top5_cands_filt, start=1):
        draw_subgraph(
            n,
            f"{group_name}: Candidate #{rank} (filtered) — {label_of_node(n)}",
            plot_dir / f"subgraph_candidate_filtered_{rank}_{nodes_df.loc[n,'GeneSymbol']}.png"
        )
    # ======================================================================
    # EXPLAINER SUBGRAPHS FOR TOP-5 FILTERED CANDIDATES
    # ======================================================================

    def build_explainer_subgraph(node_idx, edge_masks, threshold=0.5):
        """
        Constructs a subgraph using only the edges whose importance exceeds the threshold.
        """
        if len(edge_masks) == 0:
            return None

        # Average across models
        edge_importance = np.mean(np.stack(edge_masks, axis=0), axis=0)

        # Keep edges above threshold
        important_edges = np.where(edge_importance > threshold)[0].tolist()
        if len(important_edges) == 0:
            return None

        # Build subgraph edges
        edges = []
        for e_idx in important_edges:
            u = int(edges_df.loc[e_idx, "Index1"])
            v = int(edges_df.loc[e_idx, "Index2"])
            edges.append((u, v))

        return G.edge_subgraph(edges).copy()


    def draw_explainer_subgraph(node_idx, subG, title, fname):
        """
        Draws the GNNExplainer-derived computation subgraph.
        """
        if subG is None or subG.number_of_nodes() == 0:
            print(f"⚠ No explainer subgraph for node {node_idx}")
            return

        node_colors = [color_of_tag(tag_of_node(n)) for n in subG.nodes()]
        node_edges = ["#000000" if n == node_idx else "#333333" for n in subG.nodes()]

        plt.figure(figsize=(6.5, 6.5))
        pos = nx.spring_layout(subG, seed=42)
        nx.draw_networkx_nodes(subG, pos,
                            node_color=node_colors,
                            node_size=720,
                            linewidths=2,
                            edgecolors=node_edges)
        nx.draw_networkx_edges(subG, pos, alpha=0.7)
        nx.draw_networkx_labels(subG, pos,
                                labels={n: label_of_node(n) for n in subG.nodes()},
                                font_size=8,
                                font_weight="bold")

        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()


    # ---- Build and Plot Explainer Subgraphs for Top-5 Filtered Candidates ----
    for rank, n in enumerate(top5_cands_filt, start=1):

        # Load explainer masks for this candidate
        edge_masks, _ = load_masks_for_node(n, npz_files_all)

        # Build explainer subgraph
        subG = build_explainer_subgraph(n, edge_masks, threshold=0.5)

        # Plot
        draw_explainer_subgraph(
            n,
            subG,
            f"{group_name}: Explainer Subgraph #{rank} (filtered) — {label_of_node(n)}",
            plot_dir / f"explainer_subgraph_filtered_{rank}_{nodes_df.loc[n,'GeneSymbol']}.png"
        )

    # Top-5 context nodes (raw) — exclude candidates for ranking
    ctx_series_raw = pd.Series(ctx_cons_raw)
    ctx_series_raw.loc[list(INITIAL_CANDIDATE_SET)] = -np.inf
    top5_context_raw = list(ctx_series_raw.sort_values(ascending=False).head(5).index)
    for rank, n in enumerate(top5_context_raw, start=1):
        draw_subgraph(
            n,
            f"{group_name}: Context #{rank} (raw) — {label_of_node(n)}",
            plot_dir / f"subgraph_context_raw_{rank}_{nodes_df.loc[n,'GeneSymbol']}.png"
        )

    # Top-5 context nodes (filtered) — exclude candidates for ranking
    ctx_series_filt = pd.Series(ctx_cons_filt)
    ctx_series_filt.loc[list(INITIAL_CANDIDATE_SET)] = -np.inf
    top5_context_filt = list(ctx_series_filt.sort_values(ascending=False).head(5).index)
    for rank, n in enumerate(top5_context_filt, start=1):
        draw_subgraph(
            n,
            f"{group_name}: Context #{rank} (filtered) — {label_of_node(n)}",
            plot_dir / f"subgraph_context_filtered_{rank}_{nodes_df.loc[n,'GeneSymbol']}.png"
        )

    print(f"✅ Finished {group_name}. CSVs → {save_dir}, Plots → {plot_dir}")

# ======================================================================
# RUN BOTH GROUPS
# ======================================================================
if __name__ == "__main__":
    print("AAseq dim:", aaseq_dim, "NTseq dim:", ntseq_dim, "Func dim:", func_dim, "Total features:", total_features)
    run_group_pipeline("prob")
    run_group_pipeline("novelty")
    print("\n✅ All done.")
 