import warnings
warnings.filterwarnings("ignore", message="An issue occurred while importing 'pyg-lib'")
warnings.filterwarnings("ignore", message="An issue occurred while importing 'torch-sparse'")
import torch
import pandas as pd
import numpy as np
import os
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import torch.optim as optim
import networkx as nx
from torch_geometric.utils import to_networkx
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn as nn
from torch_geometric.nn import GCNConv, BatchNorm, GATConv, GATv2Conv
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Change the working directory
os.chdir('/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Scripts/')

# Load node and edge data
nodes_df = pd.read_csv('/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/CuratedData/LargestComponents/Nodes_with_properties_FC.csv')
edges_df = pd.read_csv('/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/CuratedData/LargestComponents/Edges_with_nodeproperties_FC.csv')
edges_df.insert(0, 'EdgeIndex', range(1, 1 + len(edges_df)))

# Drop unnecessary columns
nodes_df.drop(columns=['AAA', 'AMI', 'Stroke', 'HeartFailure', 'UniProt ID', 'Label'], inplace=True)

print(f"Nodes: {nodes_df.shape[0]}")
print(f"Edges: {edges_df.shape[0]}")

# Load pre-trained models
print("Loading ProtBERT model...")
#prot_tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")
#prot_model = AutoModel.from_pretrained("Rostlab/prot_bert")
print("ProtBERT model loaded successfully!")

print("Loading BioBERT model...")
#bio_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
#bio_model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
print("BioBERT model loaded successfully!")

# Embedding functions (keeping your original functions)
def get_prot_embeddings(sequence, max_length=512):
    if pd.isnull(sequence):
        return None
    inputs = prot_tokenizer(sequence, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = prot_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

def get_ntseq_embeddings(sequence, max_length=512):
    if pd.isnull(sequence):
        return None
    inputs = prot_tokenizer(sequence, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = prot_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

def get_bio_embeddings(text, max_length=512):
    if pd.isnull(text):
        return None
    max_length = min(bio_tokenizer.model_max_length, max_length)
    inputs = bio_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = bio_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

def get_label_embeddings(text, max_length=128):
    if pd.isnull(text):
        return None
    max_length = min(bio_tokenizer.model_max_length, max_length)
    inputs = bio_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = bio_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

def get_batch_embeddings(sequences, embedding_function, batch_size=32, max_length=512):
    embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        batch_embeddings = [embedding_function(seq, max_length) for seq in batch]
        if batch_embeddings:
            print(f"Shape of first embedding in batch {i // batch_size + 1}: {batch_embeddings[0].shape}")
        if any(embedding is None for embedding in batch_embeddings):
            print(f"Warning: None returned in batch {i // batch_size + 1}")
        embeddings.extend(batch_embeddings)
        print(f"Processing batch {i // batch_size + 1} of {len(sequences) // batch_size + 1}")
    return embeddings

# Prepare data
nodes_df = nodes_df.copy()
print('Processing embeddings and features...')

# Scale numeric features
scaler = StandardScaler()
numeric_features = ['Molecular Weight', 'AASEQ Length', 'NTSEQ Length']
nodes_df[numeric_features] = scaler.fit_transform(nodes_df[numeric_features])

# Convert numeric features to tensor
numeric_tensor = torch.tensor(nodes_df[numeric_features].values, dtype=torch.float)

# Load saved embeddings
aaseq_embeddings = torch.tensor(np.load('aaseq_embeddings.npy'), dtype=torch.float)
ntseq_embeddings = torch.tensor(np.load('ntseq_embeddings.npy'), dtype=torch.float)
function_embeddings = torch.tensor(np.load('function_embeddings.npy'), dtype=torch.float)

# Squeeze embeddings
aaseq_embeddings = aaseq_embeddings.squeeze(1)
ntseq_embeddings = ntseq_embeddings.squeeze(1)
function_embeddings = function_embeddings.squeeze(1)

# Concatenate all features
x = torch.cat([aaseq_embeddings, ntseq_embeddings, function_embeddings, numeric_tensor], dim=1)

# Prepare labels
labels_df = nodes_df[['PAD']].copy()  # Assuming PAD is your target column
labels_df.replace(0, -1, inplace=True)
labels = labels_df.applymap(lambda x: 1 if x == 1 else -1).values
labels = torch.tensor(labels, dtype=torch.float)

# Create masks
known_labels = labels_df.applymap(lambda x: x == 1).any(axis=1)
known_labels_indices = np.where(known_labels)[0]
train_indices = known_labels_indices
train_mask = np.zeros(len(labels_df), dtype=bool)
train_mask[train_indices] = True
test_mask = ~known_labels
test_indices = np.where(test_mask)[0]

# Process edges
edge_index = torch.tensor(edges_df[['Index1', 'Index2']].values.T, dtype=torch.long)
scores = edges_df['score'].values / 1000
edge_attr = torch.tensor(scores, dtype=torch.float).unsqueeze(1)

# Create PyTorch Geometric data object
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
embedding_dim = x.shape[1]
#
print(f"Graph created with {data.x.shape[0]} nodes and {data.edge_index.shape[1]} edges")
print(f"Feature dimension: {embedding_dim}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv
from torch_geometric.utils import negative_sampling, to_dense_adj, dense_to_sparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.manifold import TSNE

class GraphMAEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, edge_dim=1, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=num_heads, 
                             dropout=dropout, edge_dim=edge_dim)
        self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)
        
        self.gat2 = GATv2Conv(hidden_channels * num_heads, out_channels, heads=1, 
                             concat=False, dropout=dropout, edge_dim=edge_dim)
        
        self.mask_token = nn.Parameter(torch.randn(in_channels))
        
    def forward(self, x, edge_index, edge_attr=None, mask_indices=None):
        if mask_indices is not None:
            x = x.clone()
            x[mask_indices] = self.mask_token
            
        x = F.elu(self.gat1(x, edge_index, edge_attr))
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.gat2(x, edge_index, edge_attr)
        return x

class GraphMAEDecoder(nn.Module):
    def __init__(self, latent_dim, feature_dim, hidden_dim=512):
        super().__init__()
        
        self.node_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, feature_dim)
        )
        
        self.edge_decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def decode_nodes(self, z):
        return self.node_decoder(z)
    
    def decode_edges(self, z, edge_index):
        row, col = edge_index[0], edge_index[1]
        edge_embeddings = torch.cat([z[row], z[col]], dim=1)
        logits = self.edge_decoder(edge_embeddings).squeeze(-1)
        edge_pred = torch.sigmoid(logits)
        return 0.25 * edge_pred + 0.75

class GraphMAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, edge_dim=1):
        super().__init__()
        self.encoder = GraphMAEEncoder(in_channels, hidden_channels, out_channels, 
                                     num_heads, edge_dim)
        self.decoder = GraphMAEDecoder(out_channels, in_channels)
        
        self.remask_decoder = GraphMAEDecoder(out_channels, in_channels)
        
    def forward(self, x, edge_index, edge_attr=None, mask_indices=None, use_remask=False):
        z = self.encoder(x, edge_index, edge_attr, mask_indices)
        
        decoder = self.remask_decoder if use_remask else self.decoder
        
        x_recon = decoder.decode_nodes(z)
        
        edge_recon = decoder.decode_edges(z, edge_index)
        
        return z, x_recon, edge_recon

def graph_mae_loss(x_original, x_recon, edge_weights, edge_recon, mask_indices, 
                   node_weight=1.0, edge_weight=0.5, mask_weight=2.0):
    
    node_loss = F.mse_loss(x_recon, x_original, reduction='none').mean(dim=1)
    
    if mask_indices is not None and len(mask_indices) > 0:
        device = mask_indices.device
        arange_tensor = torch.arange(len(node_loss), device=device)
        mask_loss = node_loss[mask_indices].mean()
        unmask_loss = node_loss[~torch.isin(arange_tensor, mask_indices)].mean()
        total_node_loss = mask_weight * mask_loss + unmask_loss
    else:
        total_node_loss = node_loss.mean()
    
    edge_loss = F.mse_loss(edge_recon, edge_weights, reduction='mean')
    
    total_loss = node_weight * total_node_loss + edge_weight * edge_loss
    
    return total_loss, total_node_loss, edge_loss

def random_mask_nodes(num_nodes, mask_ratio=0.15):
    num_masked = int(num_nodes * mask_ratio)
    mask_indices = torch.randperm(num_nodes, device=device)[:num_masked]
    return mask_indices

def degree_based_mask_nodes(edge_index, num_nodes, mask_ratio=0.15, prefer_high_degree=True):
    degrees = torch.zeros(num_nodes, dtype=torch.long, device=device)
    degrees = degrees.scatter_add(0, edge_index[0], torch.ones_like(edge_index[0]))
    degrees = degrees.scatter_add(0, edge_index[1], torch.ones_like(edge_index[1]))
    
    num_masked = int(num_nodes * mask_ratio)
    
    if prefer_high_degree:
        _, indices = torch.topk(degrees, num_masked)
    else:
        _, indices = torch.topk(degrees, num_masked, largest=False)
    
    return indices

def adaptive_mask_nodes(x, edge_index, num_nodes, mask_ratio=0.15):
    similarities = torch.mm(F.normalize(x, p=2, dim=1), F.normalize(x, p=2, dim=1).t())
    
    neighbor_sim = torch.zeros(num_nodes, device=device)
    for i in range(num_nodes):
        neighbors = edge_index[1][edge_index[0] == i]
        if len(neighbors) > 0:
            neighbor_sim[i] = similarities[i, neighbors].mean()
    
    num_masked = int(num_nodes * mask_ratio)
    _, mask_indices = torch.topk(neighbor_sim, num_masked)
    
    return mask_indices

def contrastive_loss(z, edge_index, temperature=0.1):
    pos_pairs = edge_index.t()
    
    num_neg = pos_pairs.size(0)
    neg_pairs = torch.randint(0, z.size(0), (num_neg, 2), device=z.device)
    
    pos_sim = F.cosine_similarity(z[pos_pairs[:, 0]], z[pos_pairs[:, 1]], dim=1)
    neg_sim = F.cosine_similarity(z[neg_pairs[:, 0]], z[neg_pairs[:, 1]], dim=1)
    
    pos_loss = -torch.log(torch.sigmoid(pos_sim / temperature)).mean()
    neg_loss = -torch.log(torch.sigmoid(-neg_sim / temperature)).mean()
    
    return pos_loss + neg_loss

def train_graph_mae(data, in_channels, hidden_channels, out_channels, num_heads,
                   epochs=500, lr=1e-4, mask_ratio=0.15, masking_strategy='random',
                   use_contrastive=True, contrastive_weight=0.15,
                   save_path='/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Results/MAE/best_graph_mae_model.pth'):
    
    model = GraphMAE(in_channels, hidden_channels, out_channels, num_heads, edge_dim=1)
    model.to(device)
    
    data = data.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)
    
    best_loss = float('inf')
    patience = 100
    cnt_wait = 0
    losses = {'total': [], 'node': [], 'edge': [], 'contrastive': []}
    
    print(f"Training Graph MAE with {masking_strategy} masking strategy...")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        if masking_strategy == 'random':
            mask_indices = random_mask_nodes(data.x.size(0), mask_ratio)
        elif masking_strategy == 'degree':
            mask_indices = degree_based_mask_nodes(data.edge_index, data.x.size(0), mask_ratio)
        elif masking_strategy == 'adaptive':
            mask_indices = adaptive_mask_nodes(data.x, data.edge_index, data.x.size(0), mask_ratio)
        else:
            mask_indices = random_mask_nodes(data.x.size(0), mask_ratio)
        
        mask_indices = mask_indices.to(device)
        
        z, x_recon, edge_recon = model(data.x, data.edge_index, data.edge_attr, mask_indices)
        
        total_loss, node_loss, edge_loss = graph_mae_loss(
            data.x, x_recon, data.edge_attr.squeeze(), edge_recon, mask_indices
        )
        
        contrastive_loss_val = 0
        if use_contrastive:
            contrastive_loss_val = contrastive_loss(z, data.edge_index)
            total_loss += contrastive_weight * contrastive_loss_val
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        losses['total'].append(total_loss.item())
        losses['node'].append(node_loss.item())
        losses['edge'].append(edge_loss.item())
        losses['contrastive'].append(contrastive_loss_val.item() if use_contrastive else 0)
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            cnt_wait = 0
            torch.save(model.state_dict(), save_path)
        else:
            cnt_wait += 1
            if cnt_wait >= patience:
                print(f"Graph MAE Early stopping at epoch {epoch + 1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f'Graph MAE Epoch {epoch + 1}/{epochs}, Total: {total_loss.item():.4f}, '
                  f'Node: {node_loss.item():.4f}, Edge: {edge_loss.item():.4f}, '
                  f'Contrastive: {contrastive_loss_val.item() if use_contrastive else 0:.4f}')
    
    model.load_state_dict(torch.load(save_path))
    return model, losses

def get_graph_mae_embeddings(model, data, use_masking=False, mask_ratio=0.15):
    model.eval()
    data = data.to(device)
    
    with torch.no_grad():
        if use_masking:
            mask_indices = random_mask_nodes(data.x.size(0), mask_ratio).to(device)
            z = model.encoder(data.x, data.edge_index, data.edge_attr, mask_indices)
        else:
            z = model.encoder(data.x, data.edge_index, data.edge_attr)
        
        z = F.normalize(z, p=2, dim=1)
    
    return z.cpu().numpy()

def evaluate_reconstruction_quality(model, data, mask_ratio=0.15, num_tests=10):
    model.eval()
    data = data.to(device)
    
    reconstruction_errors = []
    
    with torch.no_grad():
        for _ in range(num_tests):
            mask_indices = random_mask_nodes(data.x.size(0), mask_ratio).to(device)
            
            z, x_recon, edge_recon = model(data.x, data.edge_index, data.edge_attr, mask_indices)
            
            if len(mask_indices) > 0:
                masked_error = F.mse_loss(x_recon[mask_indices], data.x[mask_indices], reduction='mean')
                reconstruction_errors.append(masked_error.item())
    
    return {
        'mean_reconstruction_error': np.mean(reconstruction_errors),
        'std_reconstruction_error': np.std(reconstruction_errors)
    }

def run_graph_mae_comparison(data, in_channels, hidden_channels, out_channels, num_heads,
                           epochs=300, lr=1e-4, num_runs=2):
    
    results = {}
    masking_strategies = ['random', 'degree', 'adaptive']
    
    print("="*60)
    print("TRAINING GRAPH MAE WITH DIFFERENT MASKING STRATEGIES")
    print("="*60)
    
    for strategy in masking_strategies:
        print(f"\n🔸 Training Graph MAE with {strategy} masking...")
        
        embeddings_list = []
        losses_list = []
        
        for run in range(num_runs):
            print(f"\nGraph MAE ({strategy}) Run {run + 1}/{num_runs}")
            
            model, losses = train_graph_mae(
                data, in_channels, hidden_channels, out_channels, num_heads,
                epochs=epochs, lr=lr, masking_strategy=strategy,
                save_path=f'/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Results/MAE/graph_mae_{strategy}_run_{run+1}.pth'
            )
            
            embeddings = get_graph_mae_embeddings(model, data)
            recon_quality = evaluate_reconstruction_quality(model, data)
            
            embeddings_list.append(embeddings)
            losses_list.append(losses)
            
            print(f"Reconstruction Error: {recon_quality['mean_reconstruction_error']:.4f} ± {recon_quality['std_reconstruction_error']:.4f}")
        
        results[f'GraphMAE_{strategy}'] = {
            'embeddings': np.mean(embeddings_list, axis=0),
            'embeddings_std': np.std(embeddings_list, axis=0),
            'losses': losses_list,
            'all_embeddings': embeddings_list
        }
        
        np.save(f'/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Results/MAE/graph_mae_{strategy}_embeddings.npy', 
                results[f'GraphMAE_{strategy}']['embeddings'])
        np.save(f'/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Results/MAE/graph_mae_{strategy}_embeddings_std.npy', 
                results[f'GraphMAE_{strategy}']['embeddings_std'])
    
    return results

def plot_graph_mae_comparison(results):
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    for i, (method, data) in enumerate(results.items()):
        ax = axes[0, i]
        
        for j, losses_dict in enumerate(data['losses']):
            ax.plot(losses_dict['total'], alpha=0.7, label=f'Total Run {j+1}')
            ax.plot(losses_dict['node'], alpha=0.5, linestyle='--', label=f'Node Run {j+1}')
            ax.plot(losses_dict['edge'], alpha=0.5, linestyle=':', label=f'Edge Run {j+1}')
        
        ax.set_title(f'{method} Training Losses')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    for i, (method, data) in enumerate(results.items()):
        ax = axes[1, i]
        embeddings = data['embeddings']
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) // 4))
        embeddings_2d = tsne.fit_transform(embeddings)
        
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                           alpha=0.7, s=20, cmap='viridis')
        ax.set_title(f'{method} - t-SNE Embeddings')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
    
    plt.tight_layout()
    plt.savefig('/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Results/MAE/graph_mae_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main_graph_mae():
    
    in_channels = embedding_dim
    hidden_channels = 512
    out_channels = 128
    num_heads = 16
    
    print("Starting Graph MAE training for protein network...")
    print(f"Graph: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges")
    print(f"Edge weights: {data.edge_attr.shape} (undirected, weighted)")
    
    results = run_graph_mae_comparison(
        data=data,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_heads=num_heads,
        epochs=300,
        lr=1e-4,
        num_runs=2
    )
    
    plot_graph_mae_comparison(results)
    
    print("\n" + "="*60)
    print("GRAPH MAE TRAINING COMPLETED!")
    print("="*60)
    print("🔸 Random masking: Baseline approach")
    print("🔸 Degree-based masking: Targets important nodes")
    print("🔸 Adaptive masking: Targets redundant nodes")
    print("🔸 Graph MAE learns robust representations through self-supervision")
    
    return results

if __name__ == "__main__":
    mae_results = main_graph_mae()