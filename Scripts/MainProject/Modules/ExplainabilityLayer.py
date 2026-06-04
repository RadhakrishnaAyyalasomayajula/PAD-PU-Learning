import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from torch_geometric.explain import Explainer, GNNExplainer

# ===============================================================
# CONFIG
# ===============================================================
BASE_DIR = Path("/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD")

NODES_FILE = BASE_DIR / "CuratedData/LargestComponents/Nodes_with_properties_FC.csv"
EDGES_FILE = BASE_DIR / "CuratedData/LargestComponents/Edges_with_nodeproperties_FC.csv"
CANDIDATES_FILE = BASE_DIR / "Results/ExplainabilityAnalysis/Candidates/FinalCandidates_Explainability.csv"
MODEL_DIR = BASE_DIR / "Results/ExplainabilityAnalysis/SavedGraphModels/"

# SINGLE DIRECTORY FOR ALL NPZ FILES
NPZ_DIR = BASE_DIR / "Results/ExplainabilityAnalysis/Explanations/NPZfiles/"
NPZ_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================================================
# LOAD DATA
# ===============================================================
nodes_df = pd.read_csv(NODES_FILE)
edges_df = pd.read_csv(EDGES_FILE)
edges_df.insert(0, "EdgeIndex", range(1, 1 + len(edges_df)))

nodes_df.drop(columns=["AAA","AMI","Stroke","HeartFailure","UniProt ID","Label"],
              inplace=True)

# Embeddings
aaseq = np.load(BASE_DIR / "Scripts/aaseq_embeddings.npy").squeeze(1)
ntseq = np.load(BASE_DIR / "Scripts/ntseq_embeddings.npy").squeeze(1)
func  = np.load(BASE_DIR / "Scripts/function_embeddings.npy").squeeze(1)

# Numeric features
numeric_cols = ["Molecular Weight", "AASEQ Length", "NTSEQ Length"]
nodes_df[numeric_cols] = StandardScaler().fit_transform(nodes_df[numeric_cols])
numeric = nodes_df[numeric_cols].values

# Final feature matrix
X = torch.tensor(np.concatenate([aaseq, ntseq, func, numeric], axis=1), dtype=torch.float)

# Edges
edge_index = torch.tensor(edges_df[["Index1","Index2"]].values.T, dtype=torch.long)
scores = edges_df["score"].apply(lambda x: float(np.mean(x)) if isinstance(x,(list,np.ndarray)) else float(x))
edge_attr = torch.tensor(scores.values, dtype=torch.float).view(-1,1)

graph = Data(x=X, edge_index=edge_index, edge_attr=edge_attr).to(DEVICE)

# Candidate nodes
candidate_nodes = pd.read_csv(CANDIDATES_FILE)["node_index"].astype(int).tolist()


# ===============================================================
# LOAD MODELS (EXACT ORIGINAL WORKING VERSION)
# ===============================================================
def load_encoder(model_path, input_dim):
    """
    EXACT encoder definitions that matched your saved checkpoints.
    Uses gat1/gat2 with edge_attr as originally trained.
    """
    name = model_path.name.lower()
    ckpt = torch.load(model_path, map_location=DEVICE)

    def get_state(ckpt, key=None):
        if isinstance(ckpt, dict) and key in ckpt:
            return ckpt[key]
        return ckpt

    # ----------------------------- GAE -----------------------------
    if "gae" in name and "vgae" not in name:

        class Enc(nn.Module):
            def __init__(self, in_dim):
                super().__init__()
                self.dropout = 0.5
                self.gat1 = GATv2Conv(in_dim, 512, heads=16, dropout=0.5, edge_dim=1)
                self.bn1 = nn.BatchNorm1d(512 * 16)
                self.gat2 = GATv2Conv(512 * 16, 128, heads=1, concat=False,
                                      dropout=0.5, edge_dim=1)

            def forward(self, x, edge_index, edge_attr=None):
                x = F.elu(self.gat1(x, edge_index, edge_attr))
                x = self.bn1(x)
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.gat2(x, edge_index, edge_attr)
                return x

        enc = Enc(input_dim).to(DEVICE)
        raw = get_state(ckpt, "model_state_dict")
        filtered = {k.replace("encoder.",""): v for k,v in raw.items()
                    if k.startswith("encoder.")}
        enc.load_state_dict(filtered, strict=False)
        return enc, "gae"

    # ----------------------------- VGAE -----------------------------

    if "vgae" in name:

        class Enc(nn.Module):
            """
            EXACTLY matches your trained VGAE architecture:
            - shared_encoder: GATEncoder outputs hidden_channels*2 = 1024
            - mean_layer: 1024 -> 128
            - logvar_layer: 1024 -> 128
            No gat2. No mismatched shapes.
            """
            def __init__(self, in_dim):
                super().__init__()
                hidden = 512
                heads = 16

                # This reproduces: GATEncoder(in_dim, 512, 1024, 16)
                self.shared_encoder = nn.Module()
                self.shared_encoder.gat1 = GATv2Conv(
                    in_dim, hidden, heads=heads, dropout=0.5, edge_dim=1
                )
                self.shared_encoder.bn1 = nn.BatchNorm1d(hidden * heads)

                # In your training code, GATEncoder second layer outputs 1024
                self.shared_encoder.gat2 = GATv2Conv(
                    hidden * heads, hidden * 2, heads=1,
                    concat=False, dropout=0.5, edge_dim=1
                )

                self.mean_layer = nn.Linear(hidden * 2, 128)
                self.logvar_layer = nn.Linear(hidden * 2, 128)

            def forward(self, x, edge_index, edge_attr=None):
                # Matches EXACTLY your training forward pass
                h = F.elu(self.shared_encoder.gat1(x, edge_index, edge_attr))
                h = self.shared_encoder.bn1(h)
                h = F.dropout(h, p=0.5, training=self.training)
                h = self.shared_encoder.gat2(h, edge_index, edge_attr)

                mu = self.mean_layer(h)
                logvar = self.logvar_layer(h)
                return mu  # for explainability we only need deterministic forward

        enc = Enc(input_dim).to(DEVICE)

        raw = get_state(ckpt)

        # Load ONLY real VGAE weights:
        # shared_encoder.*, mean_layer.*, logvar_layer.*
        filtered = {
            k.replace("shared_encoder.", ""): v
            for k, v in raw.items()
            if (
                k.startswith("shared_encoder.")
                or k.startswith("mean_layer")
                or k.startswith("logvar_layer")
            )
        }

        enc.load_state_dict(filtered, strict=False)
        return enc, "vgae"


    # ----------------------------- GraphMAE -----------------------------
    if "graph_mae" in name:

        class Enc(nn.Module):
            def __init__(self, in_dim):
                super().__init__()
                self.dropout = 0.5
                self.gat1 = GATv2Conv(in_dim, 512, heads=16, dropout=0.5, edge_dim=1)
                self.bn1 = nn.BatchNorm1d(512 * 16)
                self.gat2 = GATv2Conv(512 * 16, 128, heads=1, concat=False,
                                      dropout=0.5, edge_dim=1)

            def forward(self, x, edge_index, edge_attr=None):
                x = F.elu(self.gat1(x, edge_index, edge_attr))
                x = self.bn1(x)
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.gat2(x, edge_index, edge_attr)
                return x

        enc = Enc(input_dim).to(DEVICE)
        raw = get_state(ckpt)
        filtered = {k.replace("encoder.",""): v for k,v in raw.items()
                    if k.startswith("encoder.")}
        enc.load_state_dict(filtered, strict=False)
        return enc, "graphmae"

    # ----------------------------- GRACE / SimGRACE -----------------------------
    if "grace" in name or "simgrace" in name:

        class Enc(nn.Module):
            def __init__(self, in_dim):
                super().__init__()
                self.gat1 = GATv2Conv(in_dim, 256, heads=4, dropout=0.5, edge_dim=1)
                self.bn1 = nn.BatchNorm1d(256 * 4)
                self.gat2 = GATv2Conv(256 * 4, 128, heads=1, concat=False,
                                      dropout=0.5, edge_dim=1)

            def forward(self, x, edge_index, edge_attr=None):
                x = F.elu(self.gat1(x, edge_index, edge_attr))
                x = self.bn1(x)
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.gat2(x, edge_index, edge_attr)
                return x

        enc = Enc(input_dim).to(DEVICE)
        enc.load_state_dict(get_state(ckpt, "model_state_dict"))
        return enc, "grace"

    raise ValueError(f"Unrecognized model type: {model_path.name}")


# ===============================================================
# RUN EXPLANER (VERSION A + RANKING)
# ===============================================================
model_paths = sorted(list(MODEL_DIR.glob("*.pth")))
print(f"Found {len(model_paths)} runs.")

records = []

for model_path in model_paths:
    encoder, model_type = load_encoder(model_path, X.shape[1])
    encoder.eval()

    print(f"Running explanations for {model_path.name}")

    explainer = Explainer(
        model=encoder,
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="regression", task_level="node", return_type="raw")
    )

    for nid in candidate_nodes:
        explanation = explainer(graph.x, graph.edge_index, index=nid)

        edge_mask = explanation.edge_mask.detach().cpu().numpy()
        node_mask = explanation.node_mask.detach().cpu().numpy()

        # Ranking
        edge_rank = np.argsort(edge_mask)[::-1]
        node_rank = np.argsort(node_mask)[::-1]

        # Save NPZ with raw masks + rankings
        np.savez(
            NPZ_DIR / f"{model_path.stem}_node{nid}.npz",
            edge_mask=edge_mask,
            node_mask=node_mask,
            edge_rank=edge_rank,
            node_rank=node_rank
        )

        records.append({
            "model": model_path.name,
            "node_index": nid,
            "gene": nodes_df.loc[nid, "GeneSymbol"],
            "subgraph_edges": int((edge_mask > 0.5).sum())
        })

# Save metadata CSV
pd.DataFrame(records).to_csv(NPZ_DIR / "explanation_metadata.csv", index=False)

print("✅ Module 1 completed successfully.")
print(f"All NPZ files stored in:\n   {NPZ_DIR}")
