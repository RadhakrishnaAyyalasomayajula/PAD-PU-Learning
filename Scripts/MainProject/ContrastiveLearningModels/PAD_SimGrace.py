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

# ------------------ SimGRACE Modules ------------------
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

def grace_paper_loss(z1, z2, temperature=0.1):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N = z1.size(0)
    sim_matrix = torch.mm(torch.cat([z1, z2], dim=0), torch.cat([z1, z2], dim=0).T)
    sim_matrix = sim_matrix / temperature
    pos_sim = torch.diag(sim_matrix, N)
    pos_sim_rev = torch.diag(sim_matrix, -N)
    positives = torch.cat([pos_sim, pos_sim_rev], dim=0)
    mask = ~torch.eye(2 * N, dtype=torch.bool, device=z1.device)
    denom = torch.exp(sim_matrix.masked_select(mask).view(2 * N, -1)).sum(dim=1)
    numerator = torch.exp(positives)
    loss = -torch.log(numerator / denom)
    return loss.mean()

def train_simgrace(data, in_dim, hidden_dim=256, out_dim=128, epochs=300, lr=5e-4,
                   temperature=0.1, eta=0.1, noise_std=0.01,
                   save_path='best_simgrace_model.pth'):
    model = GraceGATv2Encoder(in_dim, hidden_dim, out_dim, dropout=0.3).to(device)
    projection = MLPProjection(out_dim, out_dim, out_dim).to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(projection.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_loss = float('inf')
    patience, wait = 50, 0
    losses = []

    for epoch in range(epochs):
        model.train()
        projection.train()
        optimizer.zero_grad()

        perturbed = GraceGATv2Encoder(in_dim, hidden_dim, out_dim, dropout=0.3).to(device)
        perturbed.load_state_dict(model.state_dict())
        with torch.no_grad():
            for p, orig in zip(perturbed.parameters(), model.parameters()):
                noise = torch.randn_like(p) * noise_std
                p.add_(eta * noise)

        z1 = projection(model(data.x, data.edge_index, data.edge_attr))
        z2 = projection(perturbed(data.x, data.edge_index, data.edge_attr))
        loss = grace_paper_loss(z1, z2, temperature)

        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            wait = 0
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'proj_state_dict': projection.state_dict()
            }, save_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return model, projection, losses

def get_simgrace_embeddings(model, projection, data):
    model.eval()
    projection.eval()
    data = data.to(device)
    with torch.no_grad():
        z = model(data.x, data.edge_index, data.edge_attr)
        z = projection(z)
        return F.normalize(z, p=2, dim=1).cpu().numpy()

def run_multiple_simgrace_trials(data, in_dim, hidden_dim=256, out_dim=128,
                                 runs=5, epochs=300, lr=5e-4, temperature=0.1,
                                 eta=0.1, noise_std=0.01):
    all_embeddings, all_losses = [], []
    for run in range(runs):
        print(f"\n=== SimGRACE Run {run + 1}/{runs} ===")
        model, proj, losses = train_simgrace(
            data, in_dim, hidden_dim, out_dim, epochs, lr,
            temperature, eta, noise_std,
            save_path=f"../Results/SimGRACE/best_simgrace_model_run{run+1}.pth"
        )
        mean_emb = get_simgrace_embeddings(model, proj, data)
        all_embeddings.append(mean_emb)
        all_losses.append(losses)

    avg_emb = np.mean(all_embeddings, axis=0)
    std_emb = np.std(all_embeddings, axis=0)
    return avg_emb, std_emb, all_losses

def save_embeddings(mean_emb, std_emb, model_info=None):
    out_dir = "../Results/SimGRACE"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.save(f"{out_dir}/SimGRACE_embeddings.npy", mean_emb)
    np.save(f"{out_dir}/SimGRACE_embeddings_std.npy", std_emb)

def plot_tsne(embeddings, labels=None, title="SimGRACE t-SNE Visualization", save_path="../Results/SimGRACE/tsne_plot.png"):
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) // 4))
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

def main_simgrace_multi(data, embedding_dim):
    print("=" * 60)
    print("STARTING MULTI-RUN SimGRACE TRAINING")
    print("=" * 60)

    hyperparams = {'hidden_dim': 256, 'out_dim': 128, 'epochs': 300, 'lr': 5e-4,
                   'temperature': 0.1, 'eta': 0.1, 'noise_std': 0.01, 'runs': 5}
#
    avg_emb, std_emb, all_losses = run_multiple_simgrace_trials(data, in_dim=embedding_dim, **hyperparams)
    save_embeddings(avg_emb, std_emb)
#
    plt.figure(figsize=(12, 5))
    for i, losses in enumerate(all_losses):
        plt.plot(losses, alpha=0.5, label=f'Run {i+1}')
    padded = [l + [np.nan] * (max(len(l) for l in all_losses) - len(l)) for l in all_losses]
    avg_losses = np.nanmean(padded, axis=0)
    std_losses = np.nanstd(padded, axis=0)
    plt.plot(avg_losses, label='Average', color='black', linewidth=2)
    plt.fill_between(range(len(avg_losses)), avg_losses - std_losses, avg_losses + std_losses, alpha=0.2, color='gray')
    plt.title("SimGRACE Training Loss (Multiple Runs)")
    plt.xlabel("Epoch")
    plt.ylabel("Contrastive Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    Path("../Results/SimGRACE").mkdir(parents=True, exist_ok=True)
    plt.savefig("../Results/SimGRACE/multirun_training_loss.png", dpi=300)
    plt.show()

    plot_tsne(avg_emb, title="Averaged SimGRACE Embeddings t-SNE")

    print("=" * 60)
    print("MULTI-RUN SimGRACE COMPLETED")
    print("=" * 60)
    return avg_emb, std_emb, all_losses

if __name__ == "__main__":
    avg_emb, std_emb, all_losses = main_simgrace_multi(data, embedding_dim)