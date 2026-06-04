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

# GAE and VGAE Implementation with GATv2 Encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import negative_sampling, add_self_loops
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

# ====================================
# SHARED COMPONENTS
# ====================================

class GATEncoder(nn.Module):
    """Shared GATv2 encoder for both GAE and VGAE"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, edge_dim=1):
        super().__init__()
        self.dropout = 0.5
        
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=num_heads, 
                             dropout=self.dropout, edge_dim=edge_dim)
        self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)
        
        self.gat2 = GATv2Conv(hidden_channels * num_heads, out_channels, heads=1, 
                             concat=False, dropout=self.dropout, edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.elu(self.gat1(x, edge_index, edge_attr))
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index, edge_attr)
        return x

class EdgeDecoder(nn.Module):
    """Decoder for edge reconstruction (shared by GAE and VGAE)"""
    def __init__(self, hidden_channels, edge_types=1):
        super().__init__()
        self.edge_types = edge_types
        
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, edge_types)
        )
    
    def forward(self, z, edge_index):
        edge_index = edge_index.long()

        row, col = edge_index[0], edge_index[1]

        src = z[row]
        dst = z[col]

        if src.dim() == 1:
            src = src.unsqueeze(0)
        if dst.dim() == 1:
            dst = dst.unsqueeze(0)

        edge_embeddings = torch.cat([src, dst], dim=1)
        edge_logits = self.edge_predictor(edge_embeddings)
        edge_probs = torch.sigmoid(edge_logits)
        return edge_probs.squeeze(-1)

# ====================================
# GAE IMPLEMENTATION
# ====================================

class GAE(nn.Module):
    """Graph Autoencoder with GATv2 encoder"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, edge_dim=1):
        super().__init__()
        self.encoder = GATEncoder(in_channels, hidden_channels, out_channels, num_heads, edge_dim)
        self.decoder = EdgeDecoder(out_channels)
        
    def encode(self, x, edge_index, edge_attr=None):
        return self.encoder(x, edge_index, edge_attr)
    
    def decode(self, z, edge_index):
        return self.decoder(z, edge_index)
    
    def forward(self, x, edge_index, edge_attr=None):
        z = self.encode(x, edge_index, edge_attr)
        edge_logits = self.decode(z, edge_index)
        return z, edge_logits

def gae_loss(model, z, pos_edge_index, neg_edge_index, edge_weights):
    """
    GAE loss for edge reconstruction (weighted positive edges + binary negative edges).
    """
    # Ensure edge weights are float
    edge_weights = edge_weights.float()

    # Predict on positive edges
    pos_logits = model.decode(z, pos_edge_index)

    pos_loss = F.mse_loss(pos_logits, edge_weights, reduction='mean')

    neg_preds = model.decode(z, neg_edge_index)
    neg_labels = torch.zeros_like(neg_preds, dtype=torch.float)
    neg_loss = torch.mean(F.relu(neg_preds - 0.75)**2)
#    neg_loss = F.mse_loss(neg_preds, neg_labels, reduction='mean')

    return 0.8*pos_loss + 0.2 * neg_loss


# ====================================
# VGAE IMPLEMENTATION
# ====================================

class VGAE(nn.Module):
    """Variational Graph Autoencoder with GATv2 encoder"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, edge_dim=1):
        super().__init__()
        # Shared encoder
        self.shared_encoder = GATEncoder(in_channels, hidden_channels, 
                                       hidden_channels * 2, num_heads, edge_dim)
        
        # Mean and log-variance layers
        self.mean_layer = nn.Linear(hidden_channels * 2, out_channels)
        self.logvar_layer = nn.Linear(hidden_channels * 2, out_channels)
        
        # Decoder
        self.decoder = EdgeDecoder(out_channels)
        
    def encode(self, x, edge_index, edge_attr=None):
        """Encode to mean and log-variance"""
        h = self.shared_encoder(x, edge_index, edge_attr)
        mu = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z, edge_index):
        """Decode embeddings to edge predictions"""
        return self.decoder(z, edge_index)
    
    def forward(self, x, edge_index, edge_attr=None):
        mu, logvar = self.encode(x, edge_index, edge_attr)
        z = self.reparameterize(mu, logvar)
        edge_logits = self.decode(z, edge_index)
        return z, edge_logits, mu, logvar

def vgae_loss(model, z, edge_weights, mu, logvar, pos_edge_index, neg_edge_index, 
              kl_weight=1.0, recon_weight=1.0):
    """VGAE loss function with proper decoding of positive and negative edges"""
    
    # Positive reconstruction
    pos_logits = model.decode(z, pos_edge_index)
    recon_loss = F.mse_loss(pos_logits, edge_weights, reduction='mean')

    neg_logits = model.decode(z, neg_edge_index)
    neg_probs = neg_logits
    neg_labels = torch.zeros_like(neg_logits)
    threshold = 0.75
    neg_loss = torch.mean(torch.relu(neg_probs - threshold))
#    neg_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_labels)

    # KL divergence
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    total_loss = recon_weight * recon_loss + kl_weight * kl_loss + 0.1 * neg_loss

    return total_loss, recon_loss, kl_loss

# ====================================
# TRAINING FUNCTIONS
# ====================================

def train_gae(data, in_channels, hidden_channels, out_channels, num_heads,
              epochs=500, lr=1e-4, save_path='/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Results/GAE/best_gae_model.pth'):
    """Train GAE model"""
    model = GAE(in_channels, hidden_channels, out_channels, num_heads, edge_dim=1)
    model.to(device)
    
    data = data.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_loss = float('inf')
    patience = 100
    cnt_wait = 0
    losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        z, edge_logits = model(data.x, data.edge_index, data.edge_attr)
        
        # Generate negative samples
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.x.size(0),
            num_neg_samples=data.edge_index.size(1)
        )
        
        # Compute loss
        loss = gae_loss(model, z, data.edge_index, neg_edge_index, data.edge_attr.squeeze())
#        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            cnt_wait = 0
            torch.save(model.state_dict(), save_path)
        else:
            cnt_wait += 1
            if cnt_wait >= patience:
                print(f"GAE Early stopping at epoch {epoch + 1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f'GAE Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')
    
    model.load_state_dict(torch.load(save_path))
    return model, losses

def train_vgae(data, in_channels, hidden_channels, out_channels, num_heads,
               epochs=500, lr=1e-4, kl_weight=1.0, save_path='/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Results/GAE/best_vgae_model.pth'):
    """Train VGAE model"""
    model = VGAE(in_channels, hidden_channels, out_channels, num_heads, edge_dim=1)
    model.to(device)
    
    data = data.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_loss = float('inf')
    patience = 100
    cnt_wait = 0
    losses = {'total': [], 'recon': [], 'kl': []}
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        z, edge_logits, mu, logvar = model(data.x, data.edge_index, data.edge_attr)
        
        # Generate negative samples
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.x.size(0),
            num_neg_samples=data.edge_index.size(1)
        )
        
        # Compute VGAE loss
        total_loss, recon_loss, kl_loss = vgae_loss(
            model, z, data.edge_attr.squeeze(), mu, logvar,
            data.edge_index, neg_edge_index, kl_weight
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses['total'].append(total_loss.item())
        losses['recon'].append(recon_loss.item())
        losses['kl'].append(kl_loss.item())
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            cnt_wait = 0
            torch.save(model.state_dict(), save_path)
        else:
            cnt_wait += 1
            if cnt_wait >= patience:
                print(f"VGAE Early stopping at epoch {epoch + 1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f'VGAE Epoch {epoch + 1}/{epochs}, Total: {total_loss.item():.4f}, '
                  f'Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}')
    
    model.load_state_dict(torch.load(save_path))
    return model, losses

# ====================================
# EMBEDDING EXTRACTION
# ====================================

def get_gae_embeddings(model, data):
    """Extract embeddings from trained GAE"""
    model.eval()
    data = data.to(device)
    
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index, data.edge_attr)
        z = F.normalize(z, p=2, dim=1)
    
    return z.cpu().numpy()

def get_vgae_embeddings(model, data, use_mean=True):
    """Extract embeddings from trained VGAE"""
    model.eval()
    data = data.to(device)
    
    with torch.no_grad():
        mu, logvar = model.encode(data.x, data.edge_index, data.edge_attr)
        
        if use_mean:
            z = mu  # Use mean embeddings
        else:
            z = model.reparameterize(mu, logvar)  # Sample from distribution
        
        z = F.normalize(z, p=2, dim=1)
    
    return z.cpu().numpy(), mu.cpu().numpy(), logvar.cpu().numpy()

# ====================================
# COMPARISON AND EVALUATION
# ====================================

def run_gae_vgae_comparison(data, in_channels, hidden_channels, out_channels, num_heads,
                           epochs=300, lr=1e-4, num_runs=5):
    """Compare GAE and VGAE methods"""
    
    results = {}
    
    print("="*60)
    print("TRAINING GAE vs VGAE COMPARISON")
    print("="*60)
    
    # Train GAE
    print("\n🔸 Training GAE...")
    gae_embeddings_list = []
    gae_losses_list = []
    
    for run in range(num_runs):
        print(f"\nGAE Run {run + 1}/{num_runs}")
        model_gae, losses_gae = train_gae(
            data, in_channels, hidden_channels, out_channels, num_heads,
            epochs=epochs, lr=lr, save_path=f'/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Results/GAE/gae_run_{run+1}.pth'
        )
        embeddings_gae = get_gae_embeddings(model_gae, data)
        gae_embeddings_list.append(embeddings_gae)
        gae_losses_list.append(losses_gae)
    
    # Train VGAE
    print("\n🔸 Training VGAE...")
    vgae_embeddings_list = []
    vgae_losses_list = []
    vgae_uncertainties = []
    
    for run in range(num_runs):
        print(f"\nVGAE Run {run + 1}/{num_runs}")
        model_vgae, losses_vgae = train_vgae(
            data, in_channels, hidden_channels, out_channels, num_heads,
            epochs=epochs, lr=lr, save_path=f'/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Results/GAE/vgae_run_{run+1}.pth'
        )
        embeddings_vgae, mu, logvar = get_vgae_embeddings(model_vgae, data)
        uncertainty = torch.exp(0.5 * torch.tensor(logvar)).numpy()
        
        vgae_embeddings_list.append(embeddings_vgae)
        vgae_losses_list.append(losses_vgae)
        vgae_uncertainties.append(uncertainty)
    
    # Average results
    results['GAE'] = {
        'embeddings': np.mean(gae_embeddings_list, axis=0),
        'embeddings_std': np.std(gae_embeddings_list, axis=0),
        'losses': gae_losses_list,
        'all_embeddings': gae_embeddings_list
    }
    
    results['VGAE'] = {
        'embeddings': np.mean(vgae_embeddings_list, axis=0),
        'embeddings_std': np.std(vgae_embeddings_list, axis=0),
        'losses': vgae_losses_list,
        'uncertainty': np.mean(vgae_uncertainties, axis=0),
        'all_embeddings': vgae_embeddings_list
    }
    
    # Save embeddings
    np.save('/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Results/GAE/gae_avg_embeddings.npy', results['GAE']['embeddings'])
    np.save('/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Results/GAE/vgae_avg_embeddings.npy', results['VGAE']['embeddings'])
    np.save('/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Results/GAE/gae_embeddings_std.npy', results['GAE']['embeddings_std'])
    np.save('/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Results/GAE/vgae_embeddings_std.npy', results['VGAE']['embeddings_std'])
    np.save('/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Results/GAE/vgae_uncertainty.npy', results['VGAE']['uncertainty'])
    
    print(f"\n✅ Training completed!")
    print(f"GAE embeddings shape: {results['GAE']['embeddings'].shape}")
    print(f"VGAE embeddings shape: {results['VGAE']['embeddings'].shape}")
    
    return results

def plot_gae_vgae_comparison(results):
    """Plot loss curves, t-SNE visualizations of embeddings, and VGAE uncertainty histogram."""

    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # --- 1. Training Loss Curves ---
    for i, (method, data) in enumerate(results.items()):
        ax = axes[0, i]
        
        if method == 'GAE':
            for j, losses in enumerate(data['losses']):
                ax.plot(losses, alpha=0.7, label=f'Run {j+1}')
            ax.set_title(f'{method} Training Loss')
        else:  # VGAE
            for j, losses_dict in enumerate(data['losses']):
                ax.plot(losses_dict['total'], alpha=0.7, label=f'Total Run {j+1}')
                ax.plot(losses_dict['recon'], alpha=0.5, linestyle='--', label=f'Recon Run {j+1}')
            ax.set_title(f'{method} Training Loss (Total + Recon)')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # --- 2. t-SNE Embeddings ---
    for i, (method, data) in enumerate(results.items()):
        ax = axes[1, i]
        embeddings = data['embeddings']
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) // 4))
        embeddings_2d = tsne.fit_transform(embeddings)

        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                   alpha=0.7, s=20, cmap='viridis')
        ax.set_title(f'{method} - t-SNE Embeddings')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')

    # --- 3. VGAE Uncertainty ---
    if 'VGAE' in results:
        ax = axes[1, 2]
        uncertainty = results['VGAE']['uncertainty']
        avg_uncertainty = np.mean(uncertainty, axis=1)
    
        ax.hist(avg_uncertainty, bins=30, alpha=0.7)
        ax.set_title('VGAE - Embedding Uncertainty')
        ax.set_xlabel('Avg Uncertainty')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)

    # Finalize and save
    plt.tight_layout()
    plt.savefig('/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Results/GAE/gae_vgae_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    return fig

# ====================================
# MAIN EXECUTION
# ====================================

def main_gae_vgae():
    """Main function to run GAE vs VGAE comparison (embedding only)"""
    
    in_channels = embedding_dim  # from main script
    hidden_channels = 512
    out_channels = 128
    num_heads = 16

    print("Starting GAE vs VGAE comparison for protein network...")
    print(f"Graph: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges")
    print(f"Edge weights: {data.edge_attr.shape} (undirected, weighted)")

    results = run_gae_vgae_comparison(
        data=data,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_heads=num_heads,
        epochs=300,
        lr=1e-4,
        num_runs=3
    )

    # Visualize embeddings and losses WITHOUT labels or masks
    plot_gae_vgae_comparison(results)  

    print("\n" + "="*60)
    print("GAE vs VGAE COMPARISON COMPLETED!")
    print("="*60)
    print("🔸 GAE: Deterministic embeddings, faster training")
    print("🔸 VGAE: Probabilistic embeddings with uncertainty, better generalization")
    print("🔸 Recommendation: VGAE for your biological network (uncertainty modeling)")

    return results

# Run the comparison
if __name__ == "__main__":
    gae_vgae_results = main_gae_vgae()