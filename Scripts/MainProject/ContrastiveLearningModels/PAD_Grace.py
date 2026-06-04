# ------------------ Imports & Setup ------------------
import warnings
warnings.filterwarnings("ignore", message="An issue occurred while importing 'pyg-lib'")
warnings.filterwarnings("ignore", message="An issue occurred while importing 'torch-sparse'")

import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import dropout_edge

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Data Loading ------------------
os.chdir('/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Scripts/')

nodes_df = pd.read_csv('../CuratedData/LargestComponents/Nodes_with_properties_FC.csv')
edges_df = pd.read_csv('../CuratedData/LargestComponents/Edges_with_nodeproperties_FC.csv')
edges_df.insert(0, 'EdgeIndex', range(1, 1 + len(edges_df)))
nodes_df.drop(columns=['AAA', 'AMI', 'Stroke', 'HeartFailure', 'UniProt ID', 'Label'], inplace=True)

aaseq_embeddings = torch.tensor(np.load('aaseq_embeddings.npy'), dtype=torch.float).squeeze(1)
ntseq_embeddings = torch.tensor(np.load('ntseq_embeddings.npy'), dtype=torch.float).squeeze(1)
function_embeddings = torch.tensor(np.load('function_embeddings.npy'), dtype=torch.float).squeeze(1)

scaler = StandardScaler()
numeric_features = ['Molecular Weight', 'AASEQ Length', 'NTSEQ Length']
nodes_df[numeric_features] = scaler.fit_transform(nodes_df[numeric_features])
numeric_tensor = torch.tensor(nodes_df[numeric_features].values, dtype=torch.float)

x = torch.cat([aaseq_embeddings, ntseq_embeddings, function_embeddings, numeric_tensor], dim=1)
labels_df = nodes_df[['PAD']].copy().replace(0, -1)
labels = torch.tensor(labels_df.applymap(lambda x: 1 if x == 1 else -1).values, dtype=torch.float)

edge_index = torch.tensor(edges_df[['Index1', 'Index2']].values.T, dtype=torch.long)
edge_attr = torch.tensor((edges_df['score'].values / 1000), dtype=torch.float).unsqueeze(1)

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
embedding_dim = x.shape[1]

# ------------------ GRACE Modules ------------------
class GraceGATv2Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.gat1 = GATv2Conv(in_dim, hidden_dim, heads=4, dropout=dropout, edge_dim=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 4)
        self.gat2 = GATv2Conv(hidden_dim * 4, out_dim, heads=1, concat=False, dropout=dropout, edge_dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.elu(self.gat1(x, edge_index, edge_attr))
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index, edge_attr)
        return x

class MLPProjection(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


def feature_dropout(x, drop_prob, mode='dimension'):
    if mode == 'dimension':
        drop_mask = torch.rand(x.size(1), device=x.device) > drop_prob
        return x * drop_mask.unsqueeze(0)
    else:
        drop_mask = torch.rand_like(x) > drop_prob
        return x * drop_mask.float()

def edge_dropout_with_attr(edge_index, edge_attr, drop_prob):
    if edge_attr is None:
        return dropout_edge(edge_index, p=drop_prob, training=True)[0], None
    mask = torch.rand(edge_index.size(1), device=edge_index.device) > drop_prob
    return edge_index[:, mask], edge_attr[mask]

def nt_xent_loss(z1, z2, temperature=0.1):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    sim = sim.masked_fill(torch.eye(2*N, dtype=torch.bool, device=z.device), -float('inf'))
    labels = torch.cat([torch.arange(N) + N, torch.arange(N)], dim=0).to(z.device)
    return F.cross_entropy(sim, labels)

def grace_paper_loss(z1, z2, temperature=0.1):
    """
    GRACE paper-style contrastive loss implementation.

    z1: (N, D) tensor - embeddings from view 1
    z2: (N, D) tensor - embeddings from view 2
    """
    z1 = F.normalize(z1, p=2, dim=1)  # Normalize to unit vectors
    z2 = F.normalize(z2, p=2, dim=1)
    N = z1.size(0)

    # Cosine similarity matrix
    sim_matrix = torch.mm(torch.cat([z1, z2], dim=0), torch.cat([z1, z2], dim=0).T)  # (2N, 2N)
    sim_matrix = sim_matrix / temperature

    # Positive pairs: (i, i+N) and (i+N, i)
    pos_sim = torch.diag(sim_matrix, N)  # z1[i] vs z2[i]
    pos_sim_rev = torch.diag(sim_matrix, -N)  # z2[i] vs z1[i]
    positives = torch.cat([pos_sim, pos_sim_rev], dim=0)  # (2N,)

    # Mask out positives from denominator
    mask = ~torch.eye(2 * N, dtype=torch.bool, device=z1.device)
    denom = torch.exp(sim_matrix.masked_select(mask).view(2 * N, -1)).sum(dim=1)

    # Numerators: exp(similarity of true positive)
    numerator = torch.exp(positives)

    # Final contrastive loss
    loss = -torch.log(numerator / denom)
    return loss.mean()

def train_grace(data, in_dim, hidden_dim=256, out_dim=128, epochs=300, lr=5e-4,
                feat_drop=0.2, edge_drop=0.3, temperature=0.1,
                save_path='best_grace_model.pth'):
    
    model = GraceGATv2Encoder(in_dim, hidden_dim, out_dim, dropout=0.3).to(device)
    proj = MLPProjection(out_dim, out_dim, out_dim).to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(proj.parameters()), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_loss = float('inf')
    wait, patience = 0, 50
    losses = []

    for epoch in range(epochs):
        model.train()
        proj.train()
        optimizer.zero_grad()

        ei1, ea1 = edge_dropout_with_attr(data.edge_index, data.edge_attr, edge_drop)
        ei2, ea2 = edge_dropout_with_attr(data.edge_index, data.edge_attr, edge_drop)
        x1 = feature_dropout(data.x, feat_drop, mode='dimension')
        x2 = feature_dropout(data.x, feat_drop, mode='dimension')

        z1 = proj(model(x1, ei1, ea1))
        z2 = proj(model(x2, ei2, ea2))

        loss = grace_paper_loss(z1, z2, temperature)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(proj.parameters()), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            wait = 0
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'proj_state_dict': proj.state_dict()
            }, save_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return model, proj, losses

def get_grace_embeddings(model, proj, data, passes=10, feat_drop=0.2, edge_drop=0.3):
    model.eval()
    proj.eval()
    data = data.to(device)
    embs = []

    with torch.no_grad():
        for _ in range(passes):
            ei, ea = edge_dropout_with_attr(data.edge_index, data.edge_attr, drop_prob=edge_drop)
            xd = feature_dropout(data.x, drop_prob=feat_drop, mode='dimension')
            z = proj(model(xd, ei, ea))
            embs.append(F.normalize(z, p=2, dim=1).cpu().numpy())

    embs = np.stack(embs)
    return embs.mean(0), embs.std(0)

def save_embeddings(mean_emb, std_emb, model_info=None):
    out_dir = "../Results/GRACE"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.save(f"{out_dir}/GRACE_embeddings.npy", mean_emb)
    np.save(f"{out_dir}/GRACE_embeddings_std.npy", std_emb)

def plot_tsne(embeddings, labels=None, title="GRACE t-SNE Visualization", save_path="../Results/GRACE/tsne_plot.png"):
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//4))
    emb_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    if labels is not None:
        for label in np.unique(labels):
            mask = labels == label
            plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], label=f'Label {label}', alpha=0.7)
        plt.legend()
    else:
        plt.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.7)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()

# ------------------ Multi-run GRACE ------------------
def run_multiple_grace_trials(data, in_dim, hidden_dim=256, out_dim=128, 
                              runs=5, epochs=300, lr=5e-4, feat_drop=0.2, 
                              edge_drop=0.3, temperature=0.1):
    all_embeddings, all_losses = [], []
    for run in range(runs):
        print(f"\n=== GRACE Run {run + 1}/{runs} ===")
        model, proj, losses = train_grace(
            data, in_dim, hidden_dim, out_dim, epochs, lr, feat_drop, edge_drop, temperature,
            save_path=f"../Results/GRACE/best_grace_model_run{run+1}.pth"
        )
        mean_emb, _ = get_grace_embeddings(model, proj, data, passes=1, feat_drop=feat_drop, edge_drop=edge_drop)
        all_embeddings.append(mean_emb)
        all_losses.append(losses)
    return np.mean(all_embeddings, axis=0), np.std(all_embeddings, axis=0), all_losses

def main_grace_multi():
    print("=" * 60)
    print("STARTING MULTI-RUN GRACE TRAINING")
    print("=" * 60)

    hyperparams = {'hidden_dim': 256, 'out_dim': 128, 'epochs': 300, 'lr': 5e-4,
                   'feat_drop': 0.2, 'edge_drop': 0.3, 'temperature': 0.1, 'runs': 5}

    avg_emb, std_emb, all_losses = run_multiple_grace_trials(data, in_dim=embedding_dim, **hyperparams)

    model_info = {
        'embedding_dim': embedding_dim,
        'final_embedding_dim': avg_emb.shape[1],
        'num_nodes': data.x.shape[0],
        'num_edges': data.edge_index.shape[1],
        'hyperparameters': hyperparams
    }
    save_embeddings(avg_emb, std_emb, model_info)

    plt.figure(figsize=(12, 5))
    for i, losses in enumerate(all_losses):
        plt.plot(losses, alpha=0.5, label=f'Run {i+1}')
    padded = [l + [np.nan] * (max(len(l) for l in all_losses) - len(l)) for l in all_losses]
    avg_losses = np.nanmean(padded, axis=0)
    std_losses = np.nanstd(padded, axis=0)
    plt.plot(avg_losses, label='Average', color='black', linewidth=2)
    plt.fill_between(range(len(avg_losses)), avg_losses - std_losses, avg_losses + std_losses, alpha=0.2, color='gray')
    plt.title("GRACE Training Loss (Multiple Runs)")
    plt.xlabel("Epoch")
    plt.ylabel("NT-Xent Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    Path("../Results/GRACE").mkdir(parents=True, exist_ok=True)
    plt.savefig("../Results/GRACE/multirun_training_loss.png", dpi=300)
    plt.show()

    plot_tsne(avg_emb, title="Averaged GRACE Embeddings t-SNE")

    print("=" * 60)
    print("MULTI-RUN GRACE COMPLETED")
    print("=" * 60)
    return avg_emb, std_emb, all_losses

# Run it
if __name__ == "__main__":
    avg_emb, std_emb, all_losses = main_grace_multi()