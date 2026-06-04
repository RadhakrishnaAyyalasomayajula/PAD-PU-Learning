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
edge_attr = torch.tensor(edges_df['score'].values, dtype=torch.float).unsqueeze(1)

# Create PyTorch Geometric data object
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
embedding_dim = x.shape[1]
#
print(f"Graph created with {data.x.shape[0]} nodes and {data.edge_index.shape[1]} edges")
print(f"Feature dimension: {embedding_dim}")

# ====================================
# NODE-LEVEL DGI IMPLEMENTATION
# ====================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class Discriminator(nn.Module):
    def __init__(self, feat_dim, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.hidden_layer = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.Dropout(0.3),
        )
        self.activation = nn.ReLU()
        self.f_k = nn.Bilinear(feat_dim, feat_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Bilinear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, c, h):
        h = self.activation(self.hidden_layer(h))
        return torch.sigmoid(self.f_k(h, c) / self.temperature).squeeze(-1)

class AttentionReadout(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.Tanh(),
            nn.Linear(feat_dim, 1)
        )
    
    def forward(self, x):
        att_logits = self.att(x)
        att_weights = torch.softmax(att_logits, dim=0)
        return torch.sum(att_weights * x, dim=0)

class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, edge_dim=1):
        super().__init__()
        self.dropout = 0.5
        
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=num_heads, dropout=self.dropout, edge_dim=edge_dim)
        self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)
        
        # Second layer outputs directly the final embedding dimension
        self.gat2 = GATv2Conv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=self.dropout, edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.elu(self.gat1(x, edge_index, edge_attr))
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gat2(x, edge_index, edge_attr)
        return x

class NodeDGI(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, edge_dim=1):
        super().__init__()
        self.encoder = GATEncoder(in_channels, hidden_channels, out_channels, num_heads, edge_dim)
        self.readout = AttentionReadout(out_channels)
        self.discriminator = Discriminator(out_channels)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, edge_index, edge_attr=None, corrupt_x=None):
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        
        node_emb = self.encoder(x, edge_index, edge_attr)
        graph_emb = self.readout(node_emb)

        if corrupt_x is not None:
            corrupt_emb = self.encoder(corrupt_x, edge_index, edge_attr)
            return node_emb, graph_emb, corrupt_emb
        
        return node_emb, graph_emb

def robust_graph_corruption(x, corruption_rate=0.5, noise_scale=0.15, feature_corruption_rate=0.3, shuffle_prob=0.3):
    corrupted = x.clone()
    device = x.device

    if feature_corruption_rate > 0:
        feat_mask = torch.rand_like(x) < feature_corruption_rate
        node_mask = torch.rand(x.size(0), device=device) < corruption_rate
        corrupted[node_mask] *= feat_mask[node_mask].float()

    if noise_scale > 0:
        feature_std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
        corrupted += noise_scale * feature_std * torch.randn_like(x)

    if shuffle_prob > 0 and torch.rand(1, device=device) < shuffle_prob:
        corrupted = corrupted[torch.randperm(x.size(0), device=device)]

    return corrupted

def dgi_loss(logits):
    pos_logits = logits[:, 0]
    neg_logits = logits[:, 1]
    pos_loss = F.binary_cross_entropy(pos_logits, torch.ones_like(pos_logits))
    neg_loss = F.binary_cross_entropy(neg_logits, torch.zeros_like(neg_logits))
    return (pos_loss + neg_loss) / 2

def train_node_dgi(data, in_channels, hidden_channels, out_channels, num_heads, 
                   epochs=500, lr=1e-4, save_path='best_node_dgi_model.pth'):
    model = NodeDGI(in_channels, hidden_channels, out_channels, num_heads, edge_dim=1)
    model.to(device)
    
    data = data.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.99, patience=20, verbose=True)
                                                          
    best_loss = float('inf')
    patience = 100
    cnt_wait = 0
    losses = []

    for epoch in range(epochs):
        model.train()
        
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        corrupt_x = robust_graph_corruption(x)
        
        optimizer.zero_grad()
        
        node_emb, graph_emb, corrupt_emb = model(x, edge_index, edge_attr, corrupt_x)
        
        graph_emb_exp = graph_emb.unsqueeze(0).expand(node_emb.size(0), -1)
        
        pos_logits = model.discriminator(graph_emb_exp, node_emb)
        neg_logits = model.discriminator(graph_emb_exp, corrupt_emb)
        
        logits = torch.stack([pos_logits, neg_logits], dim=1)
        loss = dgi_loss(logits)
        
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
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    return model, losses

def get_node_embeddings(model, data):
    model.eval()
    data = data.to(device)
    
    with torch.no_grad():
        node_emb, _ = model(data.x, data.edge_index, data.edge_attr)
        node_emb = F.normalize(node_emb, p=2, dim=1)
    
    return node_emb.cpu().numpy()

def run_multiple_node_dgi_trials(data, in_channels, hidden_channels, out_channels, 
                                num_heads, num_runs=5, epochs=200, lr=5e-4):
    all_embeddings = []
    all_losses = []
    
    for run in range(num_runs):
        print(f"\n=== Run {run + 1}/{num_runs} ===")
        
        model, losses = train_node_dgi(
            data, in_channels, hidden_channels, out_channels, num_heads,
            epochs=epochs, lr=lr, save_path=f'node_dgi_run_{run+1}.pth'
        )
        
        embeddings = get_node_embeddings(model, data)
        all_embeddings.append(embeddings)
        all_losses.append(losses)
        
        print(f"Run {run + 1} completed. Final embedding shape: {embeddings.shape}")
    
    avg_embeddings = np.mean(all_embeddings, axis=0)
    std_embeddings = np.std(all_embeddings, axis=0)
    
    return avg_embeddings, std_embeddings, all_losses

# ====================================
# MAIN EXECUTION
# ====================================

def main():
    # Set model parameters
    in_channels = embedding_dim
    hidden_channels = 512
    out_channels = 128
    num_heads = 16
    num_runs = 5
    
    print(f"\nStarting Node-level DGI training...")
    print(f"Graph info: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges")
    print(f"Feature dimension: {in_channels}")
    
    # Run multiple trials
    avg_embeddings, std_embeddings, all_losses = run_multiple_node_dgi_trials(
        data=data,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_heads=num_heads,
        num_runs=num_runs,
        epochs=500,
        lr=1e-4
    )
    
    print(f"\nTraining completed!")
    print(f"Average embeddings shape: {avg_embeddings.shape}")
    print(f"Standard deviation shape: {std_embeddings.shape}")
    
    # Save embeddings
    np.save('node_dgi_avg_embeddings.npy', avg_embeddings)
    np.save('node_dgi_std_embeddings.npy', std_embeddings)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    for i, losses in enumerate(all_losses):
        plt.plot(losses, alpha=0.7, label=f'Run {i+1}')
    plt.title('Training Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)

    max_len = max(len(losses) for losses in all_losses)

    # Pad shorter lists with np.nan
    padded_losses = [losses + [np.nan]*(max_len - len(losses)) for losses in all_losses]

    avg_losses = np.nanmean(padded_losses, axis=0)
    std_losses = np.std(padded_losses, axis=0)
    epochs_range = range(len(avg_losses))
    plt.plot(epochs_range, avg_losses, 'b-', label='Average')
    plt.fill_between(epochs_range, 
                     np.array(avg_losses) - np.array(std_losses),
                     np.array(avg_losses) + np.array(std_losses),
                     alpha=0.2, color='blue')
    plt.title('Average Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('node_dgi_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Now you can use the embeddings for downstream tasks
    print(f"\nEmbeddings ready for downstream tasks!")
    print(f"Training nodes: {train_mask.sum()}")
    print(f"Test nodes: {test_mask.sum()}")
    
    # Example: prepare data for classification
    X_train = avg_embeddings[train_mask]
    y_train = labels[train_mask].numpy().ravel()
    X_test = avg_embeddings[test_mask]
    y_test = labels[test_mask].numpy().ravel()
    
    print(f"Training set: {X_train.shape}, {np.unique(y_train, return_counts=True)}")
    print(f"Test set: {X_test.shape}, {np.unique(y_test, return_counts=True)}")
    
    return avg_embeddings, std_embeddings, X_train, y_train, X_test, y_test

# Run the main function
if __name__ == "__main__":
    avg_embeddings, std_embeddings, X_train, y_train, X_test, y_test = main()
    
# Load embeddings
avg_embeddings = np.load("/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Scripts/avg_embeddings.npy")
embeddings = torch.tensor(avg_embeddings, dtype=torch.float)

# Move to device if needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embeddings = embeddings.to(device)
labels = labels.to(device)