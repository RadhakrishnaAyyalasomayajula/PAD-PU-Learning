import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path("/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD")

NODES_FILE = BASE_DIR / "CuratedData/LargestComponents/Nodes_with_properties_FC.csv"
EDGES_FILE = BASE_DIR / "CuratedData/LargestComponents/Edges_with_nodeproperties_FC.csv"
NPZ_DIR    = BASE_DIR / "Results/ExplainabilityAnalysis/Explanations/NPZfiles/"
CAND_FILE  = BASE_DIR / "Results/ExplainabilityAnalysis/Candidates/FinalCandidates_Explainability.csv"

# Output folders
ELBOW_DIR   = BASE_DIR / "Results/ExplainabilityAnalysis/FeatureVsTopology_ELBOW_FINAL"
DENSITY_DIR = BASE_DIR / "Results/ExplainabilityAnalysis/FeatureVsTopology_DENSITY_FINAL"

for d in [ELBOW_DIR, DENSITY_DIR]:
    d.mkdir(parents=True, exist_ok=True)
    (d / "PerCandidatePlots").mkdir(parents=True, exist_ok=True)

# Groups from Module 2
GROUP_DIRS = {
    "prob": BASE_DIR / "Results/ExplainabilityAnalysis/Explanations/Group_prob",
    "novelty": BASE_DIR / "Results/ExplainabilityAnalysis/Explanations/Group_novelty",
}

# ============================================================
# LOAD DATA
# ============================================================
nodes_df = pd.read_csv(NODES_FILE)
edges_df = pd.read_csv(EDGES_FILE)
candidates_df = pd.read_csv(CAND_FILE)
candidates_df["group"] = ["prob"] * 50 + ["novelty"] * 50

# Full PPI graph
G = nx.from_pandas_edgelist(edges_df, "Index1", "Index2", create_using=nx.Graph())

# ============================================================
# HELPERS
# ============================================================
def list_npz_files():
    return list(NPZ_DIR.glob("*.npz"))

def load_masks_for_node(node_idx, all_npz):
    edge_masks, node_masks = [], []
    needle = f"node{node_idx}"
    for f in all_npz:
        if needle in f.stem:
            data = np.load(f)
            edge_masks.append(np.array(data["edge_mask"]))
            node_masks.append(np.array(data["node_mask"]))
    return edge_masks, node_masks

# ============================================================
# ELBOW FUNCTION
# ============================================================
def elbow_cutoff(sorted_vals):
    y = np.array(sorted_vals)
    p1 = np.array([0, y[0]])
    p2 = np.array([len(y) - 1, y[-1]])
    line_vec = p2 - p1
    norm_vec = line_vec / (np.linalg.norm(line_vec) + 1e-12)
    distances = []

    for i in range(len(y)):
        p = np.array([i, y[i]])
        proj_len = np.dot(p - p1, norm_vec)
        proj = p1 + proj_len * norm_vec
        distances.append(np.linalg.norm(p - proj))

    return max(1, int(np.argmax(distances)))

# ============================================================
# PART 1 — ELBOW SCHEME
# ============================================================
all_npz = list_npz_files()
elbow_records = []

for _, row in candidates_df.iterrows():

    node_idx = int(row["node_index"])
    group    = row["group"]
    gene     = nodes_df.loc[node_idx, "GeneSymbol"]

    edge_masks, node_masks = load_masks_for_node(node_idx, all_npz)
    if len(edge_masks) == 0: continue

    # --- feature vector ---
    cleaned = []
    for nm in node_masks:
        nm = np.array(nm)
        if nm.ndim == 2:
            cleaned.append(nm[node_idx])
        else:
            cleaned.append(nm)

    feat_vec = np.mean(np.stack(cleaned, axis=0), axis=0)
    feat_vec = np.clip(feat_vec, 0, None)
    feat_norm = feat_vec / (feat_vec.sum() + 1e-12)
    feat_sorted = np.sort(feat_norm)[::-1]

    Kf = elbow_cutoff(feat_sorted)
    FS = float(feat_sorted[:Kf+1].sum())

    # --- topology ---
    edge_avg = np.mean(np.stack(edge_masks, axis=0), axis=0)
    topo_vals = [
        max(score,0)
        for e_idx, score in enumerate(edge_avg)
        if edges_df.loc[e_idx,"Index1"]==node_idx or edges_df.loc[e_idx,"Index2"]==node_idx
    ]

    if len(topo_vals):
        topo_vals = np.array(topo_vals)
        topo_norm = topo_vals / (topo_vals.sum()+1e-12)
        topo_sorted = np.sort(topo_norm)[::-1]
        Kt = elbow_cutoff(topo_sorted)
        TS = float(topo_sorted[:Kt+1].sum())
    else:
        TS = 0.0

    total = FS + TS + 1e-12
    F_dom = FS/total
    T_dom = TS/total

    if F_dom > 1.3*T_dom: cat = "feature-driven"
    elif T_dom > 1.3*F_dom: cat = "topology-driven"
    else: cat = "balanced"

    elbow_records.append({
        "node_index": node_idx,
        "GeneSymbol": gene,
        "group": group,
        "feature_strength": FS,
        "topology_strength": TS,
        "feature_dominance": F_dom,
        "topology_dominance": T_dom,
        "category": cat
    })

    # bar plot
    plt.figure(figsize=(4,4))
    plt.bar(["Features","Topology"], [F_dom,T_dom], color=["orange","steelblue"])
    plt.ylim(0,1)
    plt.title(f"{gene} ({node_idx}) — {cat}")
    plt.tight_layout()
    plt.savefig(ELBOW_DIR / "PerCandidatePlots" / f"{node_idx}_{gene}.png")
    plt.close()

df_elbow = pd.DataFrame(elbow_records)
df_elbow.to_csv(ELBOW_DIR / "FeatureVsTopology_ElbowFinal.csv", index=False)

# ============================================================
# PART 2 — DENSITY SCHEME
# ============================================================
density_records = []

for _, row in candidates_df.iterrows():

    node_idx = int(row["node_index"])
    group    = row["group"]
    gene     = nodes_df.loc[node_idx,"GeneSymbol"]

    edge_masks, node_masks = load_masks_for_node(node_idx, all_npz)
    if len(edge_masks)==0: continue

    # feature density
    cleaned=[]
    for nm in node_masks:
        nm=np.array(nm)
        if nm.ndim==2:
            cleaned.append(nm[node_idx])
        else:
            cleaned.append(nm)
    feat_vec=np.mean(np.stack(cleaned,axis=0),axis=0)
    feat_vec=np.clip(feat_vec,0,None)
    FS=float(np.mean(feat_vec))

    # topology density
    edge_avg=np.mean(np.stack(edge_masks,axis=0),axis=0)
    incident=[
        max(score,0)
        for e_idx,score in enumerate(edge_avg)
        if edges_df.loc[e_idx,"Index1"]==node_idx or edges_df.loc[e_idx,"Index2"]==node_idx
    ]
    TS=float(np.mean(incident)) if len(incident) else 0.0

    total=FS+TS+1e-12
    F_dom=FS/total
    T_dom=TS/total

    if F_dom>1.3*T_dom: cat="feature-driven"
    elif T_dom>1.3*F_dom: cat="topology-driven"
    else: cat="balanced"

    density_records.append({
        "node_index":node_idx,
        "GeneSymbol":gene,
        "group":group,
        "feature_density":FS,
        "topology_density":TS,
        "feature_dominance":F_dom,
        "topology_dominance":T_dom,
        "category":cat
    })

    # bar plot
    plt.figure(figsize=(4,4))
    plt.bar(["Features","Topology"],[F_dom,T_dom],color=["orange","steelblue"])
    plt.ylim(0,1)
    plt.title(f"{gene} ({node_idx}) — {cat}")
    plt.tight_layout()
    plt.savefig(DENSITY_DIR/"PerCandidatePlots"/f"{node_idx}_{gene}.png")
    plt.close()

df_density=pd.DataFrame(density_records)
df_density.to_csv(DENSITY_DIR/"FeatureVsTopology_DensityFinal.csv",index=False)

# ============================================================
# SCATTER PLOTS
# ============================================================
plt.figure(figsize=(8,6))
for grp,sub in df_elbow.groupby("group"):
    plt.scatter(sub["topology_strength"],sub["feature_strength"],label=grp,s=60)
plt.xlabel("Topology Strength")
plt.ylabel("Feature Strength")
plt.title("Feature vs Topology — Elbow")
plt.legend()
plt.tight_layout()
plt.savefig(ELBOW_DIR/"Scatter_FS_TS_Elbow.png")
plt.close()

plt.figure(figsize=(8,6))
for grp,sub in df_density.groupby("group"):
    plt.scatter(sub["topology_density"],sub["feature_density"],label=grp,s=60)
plt.xlabel("Topology Density")
plt.ylabel("Feature Density")
plt.title("Feature vs Topology — Density")
plt.legend()
plt.tight_layout()
plt.savefig(DENSITY_DIR/"Scatter_FS_TS_Density.png")
plt.close()

# ============================================================
# CONSENSUS METHODS FOR TOPOLOGY
# ============================================================
def compute_group_consensus_masks(group_dir, group_name, candidates_list):
    """Compute MEAN, WEIGHTED, MAX consensus for a group."""

    stats = pd.read_csv(group_dir / "stability_unfaithfulness_pad.csv")
    stabilities = dict(zip(stats["node_index"], stats["stability"]))

    all_npz = list_npz_files()
    masks = []
    weights = []

    for nid in candidates_list:
        edge_masks, _ = load_masks_for_node(nid, all_npz)
        if len(edge_masks)==0:
            continue

        edge_avg = np.mean(np.stack(edge_masks,axis=0),axis=0)
        masks.append(edge_avg)

        s = stabilities.get(nid,0.0)
        w = 1.0/(s+1e-6)
        weights.append(w)

    if len(masks)==0:
        print(f"[WARN] No masks found for group {group_name}")
        return

    masks = np.stack(masks,axis=0)
    weights = np.array(weights)

    # MEAN
    mean_cons = masks.mean(axis=0)
    pd.DataFrame({"edge_importance": mean_cons}).to_csv(
        group_dir/"edge_importance_consensus_MEAN.csv", index=False)

    # WEIGHTED
    w_norm = weights / weights.sum()
    weighted_cons = (masks.T @ w_norm).flatten()
    pd.DataFrame({"edge_importance": weighted_cons}).to_csv(
        group_dir/"edge_importance_consensus_WEIGHTED.csv", index=False)

    # MAX
    max_cons = masks.max(axis=0)
    pd.DataFrame({"edge_importance": max_cons}).to_csv(
        group_dir/"edge_importance_consensus_MAX.csv", index=False)

    print(f"✓ Saved MEAN / WEIGHTED / MAX consensus for {group_name}")

# ============================================================
# CONSENSUS TOPOLOGY PLOTS
# ============================================================
def plot_group_topology_map(group_name, group_dir, candidates_list, variant):

    edge_file = group_dir / f"edge_importance_consensus_{variant}.csv"
    if not edge_file.exists():
        print(f"[WARN] Missing {variant} consensus for {group_name}")
        return

    df = pd.read_csv(edge_file)
    edge_imp = df["edge_importance"].values

    num_edges = len(edge_imp)
    K = max(100, int(0.01*num_edges))

    if np.allclose(edge_imp, edge_imp[0]):
        print(f"[WARN] No signal for {group_name} ({variant})")
        return

    top_idx = np.argsort(edge_imp)[::-1][:K]
    edges = []
    for e_idx in top_idx:
        u = int(edges_df.loc[e_idx,"Index1"])
        v = int(edges_df.loc[e_idx,"Index2"])
        edges.append((u,v))

    H = G.edge_subgraph(edges).copy()
    if H.number_of_nodes()==0:
        return

    candidate_set = set(candidates_list)
    known_set = set(nodes_df.index[nodes_df["PAD"]==1])

    node_colors=[]
    for n in H.nodes():
        n=int(n)
        if n in candidate_set: node_colors.append("red")
        elif n in known_set:   node_colors.append("gold")
        else:                  node_colors.append("skyblue")

    labels = {int(n): nodes_df.loc[int(n),"GeneSymbol"] for n in H.nodes()}

    plt.figure(figsize=(10,9))
    pos = nx.spring_layout(H, seed=42, k=0.35)

    nx.draw_networkx_nodes(H,pos,node_color=node_colors,node_size=500,alpha=0.95)
    nx.draw_networkx_edges(H,pos,width=1.5,alpha=0.45)
    nx.draw_networkx_labels(H,pos,labels,font_size=7,font_weight="bold")

    plt.title(f"Topology Consensus ({variant}) — {group_name}", fontsize=15)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(group_dir/f"Group_TopologyConsensus_{variant}.png", dpi=350)
    plt.close()

    print(f"✓ Saved {variant} topology for {group_name}")

# ============================================================
# FEATURE-GROUP CONSENSUS
# ============================================================
def plot_group_feature_consensus(group_name, group_dir):
    df = pd.read_csv(group_dir/"candidate_feature_groups.csv")

    groups = ["AASeq_embeddings","NTSeq_embeddings","Function_embeddings",
              "MolWeight","AASeqLen","NTSeqLen"]

    avg = df[groups].mean()

    plt.figure(figsize=(7,5))
    plt.bar(avg.index,avg.values,color="steelblue")
    plt.xticks(rotation=45,ha="right")
    plt.ylabel("Importance (mean)")
    plt.title(f"Feature Group Consensus — {group_name}")
    plt.tight_layout()
    plt.savefig(group_dir/"Group_FeatureConsensus.png",dpi=300)
    plt.close()

    print(f"✓ Saved FeatureGroupConsensus for {group_name}")

# ============================================================
# GROUP DOMINANCE BAR PLOTS
# ============================================================
def plot_group_dominance_bar(df, out_path, title):
    stats = df.groupby("group")[["feature_dominance","topology_dominance"]].mean()
    stats.to_csv(out_path.with_suffix(".csv"))

    plt.figure(figsize=(7,5))
    stats.plot(kind="bar")
    plt.ylabel("Mean Dominance (0–1)")
    plt.title(title)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"),dpi=300)
    plt.close()

plot_group_dominance_bar(
    df_elbow,
    ELBOW_DIR/"GroupDominance_BarPlot",
    "Group-Level Feature vs Topology Dominance — Elbow"
)

plot_group_dominance_bar(
    df_density,
    DENSITY_DIR/"GroupDominance_BarPlot",
    "Group-Level Feature vs Topology Dominance — Density"
)

# ============================================================
# MAIN — BUILD ALL CONSENSUS METHODS + VISUALS
# ============================================================
for grp, gdir in GROUP_DIRS.items():

    print("\n==============================")
    print(f"Building consensus for group: {grp}")
    print("==============================")

    cand_list = candidates_df[candidates_df["group"]==grp]["node_index"].astype(int).tolist()

    # 1) Compute mean + weighted + max consensus
    compute_group_consensus_masks(gdir, grp, cand_list)

    # 2) Plot MEAN, WEIGHTED, MAX
    for variant in ["MEAN","WEIGHTED","MAX"]:
        plot_group_topology_map(grp, gdir, cand_list, variant)

    # 3) Feature groups
    plot_group_feature_consensus(grp, gdir)

print("\n✓ MODULE 3 COMPLETE (Full Explainability Analysis)\n")
