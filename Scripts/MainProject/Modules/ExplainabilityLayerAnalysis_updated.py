# ======================================================================
# MODULE 2 — Final Version (Raw Only, 100 Candidates, Stability + PAD + Unfaith)
# ======================================================================

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
NPZ_DIR  = EXPL_DIR / "NPZfiles"

GROUP_PROB_DIR = EXPL_DIR / "Group_prob"
GROUP_NOV_DIR  = EXPL_DIR / "Group_novelty"

for d in [GROUP_PROB_DIR, GROUP_NOV_DIR]:
    d.mkdir(parents=True, exist_ok=True)
    (d / "CandidateSubgraphs").mkdir(parents=True, exist_ok=True)
    (d / "FeatureRawPlots").mkdir(parents=True, exist_ok=True)
    (d / "FeatureGroupPlots").mkdir(parents=True, exist_ok=True)

# ======================================================================
# LOAD DATA
# ======================================================================
nodes_df = pd.read_csv(NODES_FILE)
edges_df = pd.read_csv(EDGES_FILE)
candidates_df = pd.read_csv(CANDIDATES_FILE)

# Assign groups: first 50 prob-heavy, next 50 novelty-heavy
candidates_df["group"] = ["prob"] * 50 + ["novelty"] * 50
candidate_nodes = candidates_df["node_index"].astype(int).tolist()

# Build full graph
G = nx.from_pandas_edgelist(edges_df, "Index1", "Index2", create_using=nx.Graph())

# Known PAD labels
KNOWN_PAD = set(nodes_df.index[nodes_df["PAD"] == 1].tolist())

# ======================================================================
# FEATURE SLICES
# ======================================================================
aaseq = np.load(BASE_DIR / "Scripts/aaseq_embeddings.npy").squeeze(1)
ntseq = np.load(BASE_DIR / "Scripts/ntseq_embeddings.npy").squeeze(1)
func  = np.load(BASE_DIR / "Scripts/function_embeddings.npy").squeeze(1)

aaseq_dim, ntseq_dim, func_dim = aaseq.shape[1], ntseq.shape[1], func.shape[1]
num_numeric = 3
total_features = aaseq_dim + ntseq_dim + func_dim + num_numeric

SL_AASEQ = slice(0, aaseq_dim)
SL_NTSEQ = slice(aaseq_dim, aaseq_dim + ntseq_dim)
SL_FUNC  = slice(aaseq_dim + ntseq_dim, aaseq_dim + ntseq_dim + func_dim)

IDX_MW  = total_features - 3
IDX_AAL = total_features - 2
IDX_NTL = total_features - 1

feature_names = (
    [f"AASeq_emb_{i}" for i in range(aaseq_dim)] +
    [f"NTSeq_emb_{i}" for i in range(ntseq_dim)] +
    [f"Func_emb_{i}" for i in range(func_dim)] +
    ["MolWeight", "AASeqLen", "NTSeqLen"]
)

# UTILITIES
def list_npz_files():
    return list(NPZ_DIR.glob("*.npz"))

def load_masks_for_node(node_idx, npz_files):
    edge_masks, node_masks = [], []
    key = f"node{node_idx}"

    for f in npz_files:
        if key in f.stem:
            z = np.load(f)
            edge_masks.append(np.array(z["edge_mask"]))
            node_masks.append(np.array(z["node_mask"]))

    return edge_masks, node_masks

def stability(node_idx, npz_files):
    _, node_masks = load_masks_for_node(node_idx, npz_files)
    clean = []

    for nm in node_masks:
        nm = np.array(nm)
        if nm.ndim == 2:  # old style (N_nodes x features)
            clean.append(nm[node_idx, :])
        else:
            clean.append(nm)

    if len(clean) < 2:
        return 0.0

    arr = np.stack(clean, axis=0)
    return float(arr.var(axis=0).mean())

def unfaithfulness_pseudo(node_idx, node_mask):
    """Pseudo unfaithfulness based on mask mass removed."""
    if node_mask.sum() <= 0:
        return 0.0

    k = 10
    top_idx = np.argsort(node_mask)[-k:]
    removed = node_mask[top_idx].sum()
    total = node_mask.sum() + 1e-9
    return float(removed / total)

def pad_support(node_idx):
    neigh = list(G.neighbors(node_idx))
    if not neigh:
        return 0
    return int(nodes_df.loc[neigh, "PAD"].sum())

# ======================================================================
# VISUALIZATION HELPERS
# ======================================================================
def draw_explainer_subgraph(node_idx, edge_importance, out_file):
    th = 0.2
    important_edges = np.where(edge_importance > th)[0].tolist()

    if len(important_edges) == 0:
        return

    edges = []
    for e_idx in important_edges:
        u = int(edges_df.loc[e_idx, "Index1"])
        v = int(edges_df.loc[e_idx, "Index2"])
        edges.append((u, v))

    subG = G.edge_subgraph(edges).copy()
    if subG.number_of_nodes() == 0:
        return

    plt.figure(figsize=(7,7))
    pos = nx.spring_layout(subG, seed=42)

    # ---------------------------------------------------------
    # Color rules
    # ---------------------------------------------------------
    candidates = set(candidate_nodes)
    known_pos  = set(nodes_df.index[nodes_df["PAD"] == 1])

    node_colors = []
    for n in subG.nodes():
        n_int = int(n)
        if n_int == node_idx:
            node_colors.append("#FFA500")  # main candidate → ORANGE
        elif n_int in candidates:
            node_colors.append("#FFD700")  # other candidates → YELLOW
        elif n_int in known_pos:
            node_colors.append("#FF0000")  # known PAD → RED
        else:
            node_colors.append("#707070")  # background → DARK GREY

    # Node labels
    label_dict = {int(n): nodes_df.loc[int(n), "GeneSymbol"] for n in subG.nodes()}

    nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=650, edgecolors="black")
    nx.draw_networkx_edges(subG, pos, alpha=0.7)
    nx.draw_networkx_labels(subG, pos, labels=label_dict, font_size=8)

    # ---------------------------------------------------------
    # LEGEND
    # ---------------------------------------------------------
    import matplotlib.patches as mpatches

    legend_elements = [
        mpatches.Patch(color="#FFA500", label="Main candidate"),
        mpatches.Patch(color="#FFD700", label="Other candidates"),
        mpatches.Patch(color="#FF0000", label="Known PAD proteins"),
        mpatches.Patch(color="#707070", label="Other proteins")
    ]
    plt.legend(handles=legend_elements, loc="upper left", fontsize=8, frameon=True)

    plt.title(f"Explainer Subgraph — {nodes_df.loc[node_idx, 'GeneSymbol']}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()



def plot_feature_raw(node_idx, feat_vec, out_file):
    plt.figure(figsize=(14,4))
    plt.plot(feat_vec)
    plt.title(f"Raw Feature Importance — {nodes_df.loc[node_idx,'GeneSymbol']}")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()

def plot_feature_groups(node_idx, grouped, out_file):
    labels = list(grouped.keys())
    vals   = list(grouped.values())

    plt.figure(figsize=(7,5))
    plt.bar(labels, vals, color="steelblue")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Feature Group Importance — {nodes_df.loc[node_idx,'GeneSymbol']}")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()

# ======================================================================
# MAIN PIPELINE
# ======================================================================
def run_group(group_name, group_dir, node_indices):

    npz_files = list_npz_files()

    feats_raw_rows = []
    feats_group_rows = []
    stats_rows = []

    # consensus accumulators
    edge_sum = None
    feat_sum = None
    ctx_sum  = np.zeros(len(nodes_df), dtype=np.float64)
    count = 0

    for node_idx in node_indices:

        # -----------------------------
        # LOAD ALL EXPLANATIONS
        # -----------------------------
        edge_masks, node_masks = load_masks_for_node(node_idx, npz_files)
        if not edge_masks or not node_masks:
            continue

        edge_avg = np.mean(np.stack(edge_masks, axis=0), axis=0)

        clean = []
        for nm in node_masks:
            nm = np.array(nm)
            if nm.ndim == 2:
                clean.append(nm[node_idx, :])
            else:
                clean.append(nm)

        feat_avg = np.mean(np.stack(clean, axis=0), axis=0)

        # -----------------------------
        # CONSENSUS ACCUMULATION
        # -----------------------------
        if edge_sum is None:
            edge_sum = np.zeros_like(edge_avg, float)
        if feat_sum is None:
            feat_sum = np.zeros_like(feat_avg, float)

        edge_sum += edge_avg
        feat_sum += feat_avg
        count += 1

        # context accumulation
        for e_idx, score in enumerate(edge_avg):
            u = edges_df.loc[e_idx, "Index1"]
            v = edges_df.loc[e_idx, "Index2"]
            ctx_sum[u] += score
            ctx_sum[v] += score

        # -----------------------------
        # PER-CANDIDATE METRICS
        # -----------------------------
        stab = stability(node_idx, npz_files)
        unfaith = unfaithfulness_pseudo(node_idx, feat_avg)
        pad_sup = pad_support(node_idx)

        stats_rows.append({
            "node_index": node_idx,
            "GeneSymbol": nodes_df.loc[node_idx, "GeneSymbol"],
            "stability": stab,
            "unfaithfulness": unfaith,
            "pad_support": pad_sup
        })

        # -----------------------------
        # FEATURE GROUPS
        # -----------------------------
        grouped = {
            "AASeq_embeddings": float(feat_avg[SL_AASEQ].sum()),
            "NTSeq_embeddings": float(feat_avg[SL_NTSEQ].sum()),
            "Function_embeddings": float(feat_avg[SL_FUNC].sum()),
            "MolWeight": float(feat_avg[IDX_MW]),
            "AASeqLen": float(feat_avg[IDX_AAL]),
            "NTSeqLen": float(feat_avg[IDX_NTL]),
        }

        # save raw row
        row = {"node_index": node_idx, "GeneSymbol": nodes_df.loc[node_idx, "GeneSymbol"]}
        for i, v in enumerate(feat_avg):
            row[f"f_{i}"] = float(v)
        feats_raw_rows.append(row)

        # save grouped row
        g_row = {"node_index": node_idx, "GeneSymbol": nodes_df.loc[node_idx, "GeneSymbol"]}
        g_row.update(grouped)
        feats_group_rows.append(g_row)

        # -----------------------------
        # PLOTS
        # -----------------------------
        plot_feature_raw(node_idx, feat_avg,
                         group_dir / "FeatureRawPlots" / f"{node_idx}_{nodes_df.loc[node_idx,'GeneSymbol']}.png")

        plot_feature_groups(node_idx, grouped,
                            group_dir / "FeatureGroupPlots" / f"{node_idx}_{nodes_df.loc[node_idx,'GeneSymbol']}.png")

        # -----------------------------
        # EXPLAINER SUBGRAPH
        # -----------------------------
        draw_explainer_subgraph(
            node_idx,
            edge_avg,
            group_dir / "CandidateSubgraphs" / f"{node_idx}_{nodes_df.loc[node_idx,'GeneSymbol']}.png"
        )

    # ======================================================================
    # SAVE CONSENSUS
    # ======================================================================
    edge_cons = edge_sum / count
    feat_cons = feat_sum / count
    ctx_cons  = ctx_sum / count

    pd.DataFrame({"edge_importance": edge_cons}).to_csv(group_dir / "edge_importance_consensus.csv", index=False)

    pd.Series(feat_cons, index=feature_names, name="feature_importance") \
      .to_csv(group_dir / "feature_importance_consensus.csv")

    pd.Series(ctx_cons, index=nodes_df.index, name="context_importance") \
      .to_csv(group_dir / "context_importance_consensus.csv")

    # ======================================================================
    # SAVE PER-CANDIDATE FILES
    # ======================================================================
    pd.DataFrame(stats_rows).to_csv(group_dir / "stability_unfaithfulness_pad.csv", index=False)
    pd.DataFrame(feats_raw_rows).to_csv(group_dir / "candidate_feature_importance.csv", index=False)
    pd.DataFrame(feats_group_rows).to_csv(group_dir / "candidate_feature_groups.csv", index=False)

    print(f"✓ Finished group {group_name} — results saved to {group_dir}")


# ======================================================================
# RUN BOTH GROUPS
# ======================================================================
if __name__ == "__main__":
    prob_nodes = candidates_df[candidates_df["group"] == "prob"]["node_index"].astype(int).tolist()
    nov_nodes  = candidates_df[candidates_df["group"] == "novelty"]["node_index"].astype(int).tolist()

    run_group("Probability-heavy", GROUP_PROB_DIR, prob_nodes)
    run_group("Novelty-heavy", GROUP_NOV_DIR, nov_nodes)

    print("\n✓ MODULE 2 COMPLETE\n")
