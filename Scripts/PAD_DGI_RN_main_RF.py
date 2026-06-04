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
#

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

#    
# Load embeddings
avg_embeddings = np.load("/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Scripts/node_dgi_avg_embeddings.npy")
embeddings = torch.tensor(avg_embeddings, dtype=torch.float)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embeddings = embeddings.to(device)
labels = labels.to(device)
embedding_dim = embeddings.shape[1]

#
pos_indices = (labels == 1).nonzero(as_tuple=True)[0]
unlabeled_indices = (labels == -1).nonzero(as_tuple=True)[0]

pos_embs = embeddings[pos_indices]         
unl_embs = embeddings[unlabeled_indices]   

dist_matrix = torch.cdist(unl_embs, pos_embs)  
min_distances, _ = torch.min(dist_matrix, dim=1)  

sorted_distances, sorted_indices = torch.sort(min_distances, descending=True)

max_rn = 68
rn_count = min(max_rn, len(unlabeled_indices))
rn_indices = unlabeled_indices[sorted_indices[:rn_count]]

labels[rn_indices] = 0

train_indices = torch.cat([pos_indices, rn_indices])

remaining_unlabeled_mask = torch.ones(len(unlabeled_indices), dtype=torch.bool)
remaining_unlabeled_mask[sorted_indices[:rn_count]] = False
test_indices = unlabeled_indices[remaining_unlabeled_mask]

train_mask = torch.zeros(len(labels), dtype=torch.bool, device=device)
train_mask[train_indices] = True

test_mask = torch.zeros(len(labels), dtype=torch.bool, device=device)
test_mask[test_indices] = True

print(f"Number of positives: {len(pos_indices)}")
print(f"Number of reliable negatives selected: {len(rn_indices)}")
print(f"Number of test samples: {len(test_indices)}")
#
#
import torch
import torch.nn as nn

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           average_precision_score, precision_recall_curve)
import torch
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           average_precision_score, precision_recall_curve)
from copy import deepcopy

def run_random_forest_pu_learning(X, initial_labels, gene_symbols, 
                                 rf_params_rn=None, rf_params_final=None, 
                                 cv_folds=5, reliable_negative_ratio=25.0):
    """
    Perform Random Forest PU Learning for biological data
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix (embeddings)
    initial_labels : numpy.ndarray  
        Initial labels (1 for positive, -1 for unlabeled)
    gene_symbols : numpy.ndarray
        Gene symbols for identification
    rf_params_rn : dict
        Random Forest parameters for reliable negative selection
    rf_params_final : dict
        Random Forest parameters for final classification
    cv_folds : int
        Number of cross-validation folds
    reliable_negative_ratio : float
        Ratio of reliable negatives to positives
    
    Returns:
    --------
    dict containing results, models, and predictions
    """
    
    print("Starting Random Forest PU Learning...")
    
    # Default parameters if not provided
    if rf_params_rn is None:
        rf_params_rn = {
            'n_estimators': 300,
            'max_depth': 10,
            'min_samples_split': 8,
            'min_samples_leaf': 4,
            'max_features': 'sqrt',
            'bootstrap': True,
            'class_weight': {0: 1, 1: 5},
            'random_state': 42,
            'n_jobs': -1,
            'oob_score': True
        }
    
    if rf_params_final is None:
        rf_params_final = {
            'n_estimators': 500,
            'max_depth': 12,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1,
            'oob_score': True
        }
    
    # Identify positive and unlabeled samples
    pos_indices = np.where(initial_labels == 1)[0]
    unlabeled_indices = np.where(initial_labels == -1)[0]
    
    print(f"Positive samples: {len(pos_indices)}")
    print(f"Unlabeled samples: {len(unlabeled_indices)}")
    
    # Step 1: Train initial model on positive vs unlabeled
    print("\nStep 1: Training initial RF model for reliable negative selection...")
    
    # Prepare data for RN selection (P vs U)
    X_pu = np.vstack([X[pos_indices], X[unlabeled_indices]])
    y_pu = np.hstack([np.ones(len(pos_indices)), np.zeros(len(unlabeled_indices))])
    
    # Train RF for reliable negative selection
    rf_rn = RandomForestClassifier(**rf_params_rn)
    rf_rn.fit(X_pu, y_pu)
    
    # Get probabilities for unlabeled samples
    unlabeled_probs = rf_rn.predict_proba(X[unlabeled_indices])[:, 0]  # Probability of being negative
    
    # Select reliable negatives (those with highest probability of being negative)
    n_reliable_negatives = int(len(pos_indices) * reliable_negative_ratio)
    n_reliable_negatives = min(n_reliable_negatives, len(unlabeled_indices))
    
    # Sort by probability of being negative (descending)
    rn_candidates = np.argsort(unlabeled_probs)[::-1]
    selected_rn_indices = unlabeled_indices[rn_candidates[:n_reliable_negatives]]
    
    print(f"Selected {len(selected_rn_indices)} reliable negatives")
    print(f"RN selection threshold probability: {unlabeled_probs[rn_candidates[n_reliable_negatives-1]]:.4f}")
    
    # Step 2: Train final model on P + RN
    print("\nStep 2: Training final RF model on P + RN...")
    
    # Prepare final training data
    final_train_indices = np.hstack([pos_indices, selected_rn_indices])
    X_final = X[final_train_indices]
    y_final = np.hstack([np.ones(len(pos_indices)), np.zeros(len(selected_rn_indices))])
    
    rf_final = RandomForestClassifier(**rf_params_final)
    rf_final.fit(X_final, y_final)
    
    print("\nStep 3: Performing cross-validation...")
    
    cv_results = []
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_final, y_final)):
        print(f"  Processing fold {fold + 1}/{cv_folds}")
        
        X_train_fold = X_final[train_idx]
        y_train_fold = y_final[train_idx]
        X_val_fold = X_final[val_idx]
        y_val_fold = y_final[val_idx]
        
        # Train fold model
        rf_fold = RandomForestClassifier(**rf_params_final)
        rf_fold.fit(X_train_fold, y_train_fold)
        
        # Predictions
        y_pred_fold = rf_fold.predict(X_val_fold)
        y_prob_fold = rf_fold.predict_proba(X_val_fold)[:, 1]
        
        # Calculate metrics
        fold_results = {
            'fold': fold + 1,
            'accuracy': accuracy_score(y_val_fold, y_pred_fold),
            'precision': precision_score(y_val_fold, y_pred_fold, zero_division=0),
            'recall': recall_score(y_val_fold, y_pred_fold, zero_division=0),
            'f1': f1_score(y_val_fold, y_pred_fold, zero_division=0),
            'auc_roc': roc_auc_score(y_val_fold, y_prob_fold) if len(np.unique(y_val_fold)) > 1 else 0,
            'auc_pr': average_precision_score(y_val_fold, y_prob_fold) if len(np.unique(y_val_fold)) > 1 else 0
        }
        cv_results.append(fold_results)
    
    # Calculate average CV results
    avg_cv_results = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr']:
        values = [result[metric] for result in cv_results]
        avg_cv_results[f'{metric}_mean'] = np.mean(values)
        avg_cv_results[f'{metric}_std'] = np.std(values)
    
    print("\nCross-validation Results:")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr']:
        mean_val = avg_cv_results[f'{metric}_mean']
        std_val = avg_cv_results[f'{metric}_std']
        print(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Step 4: Predict on all samples
    print("\nStep 4: Making predictions on all samples...")
    
    all_predictions = rf_final.predict(X)
    all_probabilities = rf_final.predict_proba(X)[:, 1]
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'GeneSymbol': gene_symbols,
        'Original_Label': initial_labels,
        'Prediction': all_predictions,
        'Prediction_Probability': all_probabilities,
        'Is_Training_Positive': np.isin(np.arange(len(gene_symbols)), pos_indices),
        'Is_Selected_RN': np.isin(np.arange(len(gene_symbols)), selected_rn_indices),
        'Is_Remaining_Unlabeled': np.isin(np.arange(len(gene_symbols)), 
                                        np.setdiff1d(unlabeled_indices, selected_rn_indices))
    })
    
    # Add confidence levels
    def assign_confidence(prob):
        if prob >= 0.8:
            return 'High_Positive'
        elif prob >= 0.6:
            return 'Medium_Positive'
        elif prob >= 0.4:
            return 'Low_Confidence'
        elif prob >= 0.2:
            return 'Medium_Negative'
        else:
            return 'High_Negative'
    
    results_df['Confidence_Level'] = results_df['Prediction_Probability'].apply(assign_confidence)
    
    # Summary statistics
    print(f"\nPrediction Summary:")
    print(f"  High confidence positives (≥0.8): {len(results_df[results_df['Prediction_Probability'] >= 0.8])}")
    print(f"  Medium confidence positives (0.6-0.8): {len(results_df[(results_df['Prediction_Probability'] >= 0.6) & (results_df['Prediction_Probability'] < 0.8)])}")
    print(f"  Low confidence (0.4-0.6): {len(results_df[(results_df['Prediction_Probability'] >= 0.4) & (results_df['Prediction_Probability'] < 0.6)])}")
    print(f"  Medium confidence negatives (0.2-0.4): {len(results_df[(results_df['Prediction_Probability'] >= 0.2) & (results_df['Prediction_Probability'] < 0.4)])}")
    print(f"  High confidence negatives (<0.2): {len(results_df[results_df['Prediction_Probability'] < 0.2])}")
    
    return {
        'results_df': results_df,
        'learner': rf_rn,  # RN selection model
        'final_model': rf_final,  # Final classification model
        'cv_results': cv_results,
        'avg_cv_results': avg_cv_results,
        'selected_rn_indices': selected_rn_indices,
        'pos_indices': pos_indices,
        'unlabeled_indices': unlabeled_indices,
        'rn_selection_probs': unlabeled_probs
    }

# Using Random Forest PU Learning the same way as XGBoost
X = embeddings.cpu().numpy()  
initial_labels = labels.squeeze().cpu().numpy()  
gene_symbols = nodes_df['GeneSymbol'].values  

rf_results = run_random_forest_pu_learning(
    X, initial_labels, gene_symbols,
    rf_params_rn=None,
    rf_params_final=None
)

# Extract results dataframe (same structure as XGBoost)
rf_results_df = rf_results['results_df']
print("Random Forest Results:")
print(rf_results_df.head(10))

# Add NodeIndex if needed (to match your XGBoost workflow)
rf_results_df['NodeIndex'] = range(len(rf_results_df))

# Group by NodeIndex and GeneSymbol (same as XGBoost)
rf_results_df = rf_results_df.groupby(["NodeIndex", "GeneSymbol"]).first().reset_index()

# Save results to Excel (same format as XGBoost)
rf_results_df.to_excel(
    '/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Results/DGI/RandomForest_combined.xlsx', 
    index=False
)

print(f"Random Forest results saved with {len(rf_results_df)} samples")