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

##############################################################################################
#    Asymmetric Positive Class (y=1) Unified Focal loss with GEV link for XGBoost         #
##############################################################################################

def asymmetric_unified_focal_gev1(alpha=1.0, delta=0.7, gamma1=0.0, gamma2=1.0, tau=-0.25, pi=0.5):    
    def aUF_gev(preds: np.ndarray, dtrain: xgb.DMatrix):
        """
        Combined asymmetric focal loss with GEV link for extreme imbalance
        """
        # Add numerical stability
        eps = 1e-7
        preds = np.clip(preds, eps, 1-eps)
        
        # Domain check for GEV transformation
        if tau < 0:
            max_pred = -1/tau - eps
            preds = np.clip(preds, -np.inf, max_pred)
            
        grad_maf_gev = gradient_maf_gev1(preds, dtrain, alpha, gamma1, tau)
        grad_mft_gev = gradient_mft_gev(preds, dtrain, delta, gamma2, tau)
        
        hess_maf_gev = hessian_maf_gev1(preds, dtrain, alpha, gamma1, tau)
        hess_mft_gev = hessian_mft_gev(preds, dtrain, delta, gamma2, tau)
        
        grad_auf_gev = (pi * grad_maf_gev) + ((1-pi) * grad_mft_gev)
        hess_auf_gev = (pi * hess_maf_gev) + ((1-pi) * hess_mft_gev)
        
        return grad_auf_gev, hess_auf_gev
    return aUF_gev

def gradient_mft_gev(preds: np.ndarray, dtrain: xgb.DMatrix, delta=0.7, gamma2=1.0, tau=-0.25):           
    labels = dtrain.get_label()
    eps = 1e-7
    preds = np.clip(preds, eps, 1-eps)
    
    # GEV transformation with stability
    gev_preds = np.exp(-np.power(np.maximum(1 + (tau*preds), eps), -1/tau))
    gev_preds = np.clip(gev_preds, eps, 1-eps)
    
    phi1 = gev_preds*(labels+delta-1) - (delta*labels)
    phi2 = gev_preds*(delta-1) - (delta*labels)
    
    # Avoid division by zero
    phi1 = np.where(np.abs(phi1) < eps, eps * np.sign(phi1), phi1)
    phi2 = np.where(np.abs(phi2) < eps, eps * np.sign(phi2), phi2)
    
    L1_mF = -np.divide((delta*labels)*np.power(np.abs(phi1/phi2), 1/gamma2), 
                       gamma2*phi1*phi2)
    
    # GEV derivative with stability
    log_term = np.maximum(-np.log(np.maximum(gev_preds, eps)), eps)
    P1_gev = gev_preds * np.power(log_term, tau+1)
    
    grad = L1_mF * P1_gev  
    return grad

def hessian_mft_gev(preds: np.ndarray, dtrain: xgb.DMatrix, delta=0.7, gamma2=1.0, tau=-0.25):      
    labels = dtrain.get_label()
    eps = 1e-7
    preds = np.clip(preds, eps, 1-eps)
    
    gev_preds = np.exp(-np.power(np.maximum(1 + (tau*preds), eps), -1/tau))
    gev_preds = np.clip(gev_preds, eps, 1-eps)
    
    phi1 = gev_preds*(labels+delta-1) - (delta*labels)
    phi2 = gev_preds*(delta-1) - (delta*labels)
    
    phi1 = np.where(np.abs(phi1) < eps, eps * np.sign(phi1), phi1)
    phi2 = np.where(np.abs(phi2) < eps, eps * np.sign(phi2), phi2)
    
    L1_mF = -np.divide((delta*labels)*np.power(np.abs(phi1/phi2), 1/gamma2), 
                       gamma2*phi1*phi2)
    L2_mF = np.divide((delta*labels)*np.power(np.abs(phi1/phi2), 1/gamma2)*
                      ((2*(delta-1)*gamma2*phi1)-(labels*delta*(gamma2-1))),
                      np.power(gamma2*phi1*phi2, 2))
    
    log_term = np.maximum(-np.log(np.maximum(gev_preds, eps)), eps)
    P1_gev = gev_preds * np.power(log_term, tau+1)
    P2_gev = -gev_preds*((tau+1)*(np.power(log_term, -1))-1)*np.power(log_term, 2*(tau+1))
    
    hess = (L1_mF*P2_gev) + ((P1_gev**2)*L2_mF)
    return hess

def gradient_maf_gev1(preds: np.ndarray, dtrain: xgb.DMatrix, alpha=1.0, gamma1=0.0, tau=-0.25):           
    labels = dtrain.get_label()
    eps = 1e-7
    preds = np.clip(preds, eps, 1-eps)
    
    gev_preds = np.exp(-np.power(np.maximum(1 + (tau*preds), eps), -1/tau))
    gev_preds = np.clip(gev_preds, eps, 1-eps)
    
    # Asymmetric focal components with stability
    one_minus_p = np.maximum(1-gev_preds, eps)
    log_p = np.log(np.maximum(gev_preds, eps))
    
    n1s = -(gamma1*gev_preds*np.power(one_minus_p, gamma1-1)*log_p) + np.power(one_minus_p, gamma1)
    n2a = (one_minus_p*np.log(one_minus_p)) - gev_preds       
    
    L1_maF = -((alpha*labels*n1s*one_minus_p) + ((1-labels)*gev_preds*n2a))/(gev_preds*one_minus_p)
    
    log_term = np.maximum(-np.log(np.maximum(gev_preds, eps)), eps)
    P1_gev = gev_preds * np.power(log_term, tau+1)
    
    grad = L1_maF * P1_gev  
    return grad

def hessian_maf_gev1(preds: np.ndarray, dtrain: xgb.DMatrix, alpha=1.0, gamma1=0.0, tau=-0.25):      
    labels = dtrain.get_label()
    eps = 1e-7
    preds = np.clip(preds, eps, 1-eps)
    
    gev_preds = np.exp(-np.power(np.maximum(1 + (tau*preds), eps), -1/tau))
    gev_preds = np.clip(gev_preds, eps, 1-eps)
    
    one_minus_p = np.maximum(1-gev_preds, eps)
    log_p = np.log(np.maximum(gev_preds, eps))
    
    n1s = -(gamma1*gev_preds*np.power(one_minus_p, gamma1-1)*log_p) + np.power(one_minus_p, gamma1)
    n2a = (one_minus_p*np.log(one_minus_p)) - gev_preds
    
    # Second derivatives (simplified for stability)
    n3s = -(2*gamma1*gev_preds*np.power(one_minus_p, gamma1-1)) - np.power(one_minus_p, gamma1) + \
          (np.power(gamma1*gev_preds, 2)*np.power(one_minus_p, gamma1-2)*log_p) - \
          (gamma1*gev_preds*gev_preds*np.power(one_minus_p, gamma1-2)*log_p)
    n4a = -one_minus_p      
    
    L1_maF = -((alpha*labels*n1s*one_minus_p) + ((1-labels)*gev_preds*n2a))/(gev_preds*one_minus_p)
    L2_maF = -((alpha*labels*n3s*np.power(one_minus_p, 2)) + ((1-labels)*np.power(gev_preds, 2)*n4a))/np.power(gev_preds*one_minus_p, 2)
    
    log_term = np.maximum(-np.log(np.maximum(gev_preds, eps)), eps)
    P1_gev = gev_preds * np.power(log_term, tau+1)
    P2_gev = -gev_preds*((tau+1)*(np.power(log_term, -1))-1)*np.power(log_term, 2*(tau+1))
    
    hess = (L1_maF*P2_gev) + (np.power(P1_gev, 2)*L2_maF)
    return hess

import numpy as np
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, average_precision_score, confusion_matrix)
from sklearn.model_selection import StratifiedKFold

def specificity_eval(y_pred, y_true):
    """
    Custom XGBoost evaluation function for specificity
    
    Args:
        y_pred: predicted probabilities
        y_true: DMatrix with true labels
    
    Returns:
        eval_name, eval_result
    """
    # Get true labels from DMatrix
    labels = y_true.get_label()
    
    # Convert probabilities to binary predictions
    predictions = (y_pred >= 0.5).astype(int)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    
    # Calculate specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return 'specificity', specificity, True


class XGBoostPULearner:
    def __init__(self, loss_params=None, xgb_params=None):
        """
        XGBoost-based PU learner with custom asymmetric focal loss
        
        Args:
            loss_params: dict with keys alpha, delta, gamma1, gamma2, tau, pi
            xgb_params: dict with XGBoost parameters
        """
        self.loss_params = loss_params or {
            'alpha': 1,      
            'delta': 0.6,     
            'gamma1': 0.8,     
            'gamma2': 2.0,     
            'tau': -0.25,      
            'pi': 0.6          
        }
        
        self.loss_params_phase1 = {
            'alpha': 1.5,
            'delta': 0.8,
            'gamma1': 0.2,
            'gamma2': 3.00,
            'tau': -0.25,
            'pi': 0.2
        }       

        self.loss_params_phase2 = {
            'alpha': 1.0,
            'delta': 0.75,
            'gamma1': 2.5,
            'gamma2': 0.25,
            'tau': -0.25,
            'pi': 0.9
        }

        self.xgb_params_rn = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'learning_rate': 0.05,
            'max_depth': 5,
            'min_child_weight': 3,
            'subsample': 0.85,
            'colsample_bytree': 0.9,
            'gamma': 2,                        
            'lambda': 1.0,                     
            'alpha': 0.5,  
            #'scale_pos_weight': 0.5 * 200,
            'tree_method': 'approx',
            'n_estimators': 100
        }
        
        self.xgb_params_final = {
            'objective': 'binary:logistic',
            'eval_metric': ['auc'],  
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 5,
            'subsample': 0.85,
            'colsample_bytree': 0.9,
            'gamma': 5,                        
            'lambda': 2.0,                     
            'alpha': 0.5,  
            #'scale_pos_weight': 0.5 * 2,
            'tree_method': 'approx',
            'n_estimators': 50
        }
        
        self.model = None
        self.rn_identification_history = []
        
    def iterative_rn_identification(self, X, initial_labels, n_iterations=5, 
                                  initial_threshold=0.75, threshold_decay=0.025, 
                                  min_threshold=0.75, early_stopping_rounds=50):
        """
        Phase 1: Iteratively identify Reliable Negatives using XGBoost
        """
        current_labels = initial_labels.copy()
        current_threshold = initial_threshold
        
        print("="*70)
        print("PHASE 1: ITERATIVE RELIABLE NEGATIVE IDENTIFICATION (XGBoost)")
        print("="*70)
        
        initial_positives = np.sum(current_labels == 1)
        initial_rns = np.sum(current_labels == 0) 
        initial_unlabeled = np.sum(current_labels == -1)
        
        print(f"Starting with:")
        print(f"  Positives: {initial_positives}")
        print(f"  Initial RNs: {initial_rns}")
        print(f"  Unlabeled: {initial_unlabeled}")
        
        for iteration in range(n_iterations):
            print(f"\n--- RN Identification Iteration {iteration + 1}/{n_iterations} ---")
            print(f"Current threshold: {current_threshold:.3f}")
            
            train_mask = current_labels != -1
            X_train = X[train_mask]
            y_train = current_labels[train_mask]
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            
            # custom_loss = asymmetric_unified_focal_gev1(**self.loss_params_phase1)
            
            model = xgb.train(
                params=self.xgb_params_rn,
                dtrain=dtrain,
                num_boost_round=self.xgb_params_rn.get('n_estimators', 100),
                #obj=custom_loss,
                #early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False
            )
            
            dall = xgb.DMatrix(X)
            pred_scores = model.predict(dall)
            
            unlabeled_mask = current_labels == -1
            unlabeled_indices = np.where(unlabeled_mask)[0]
            
            if len(unlabeled_indices) == 0:
                print("No more unlabeled nodes to process!")
                break
                
            unlabeled_scores = pred_scores[unlabeled_indices]
            
            negative_threshold = 1 - current_threshold
            new_rn_mask = unlabeled_scores <= negative_threshold
            new_rn_indices = unlabeled_indices[new_rn_mask]
            
            max_total_rns = 3000  

            current_rns = np.sum(current_labels == 0)
            remaining_slots = max_total_rns - current_rns

            if remaining_slots <= 0:
                print(f"Reached total RN cap of {max_total_rns}. Stopping RN identification.")
                break

            if len(new_rn_indices) > remaining_slots:
                sorted_idx = np.argsort(unlabeled_scores[new_rn_mask])
                selected_idx = sorted_idx[:remaining_slots]
                new_rn_indices = new_rn_indices[selected_idx]

            if len(new_rn_indices) > 0:
                current_labels[new_rn_indices] = 0
                
            current_positives = np.sum(current_labels == 1)
            current_rns = np.sum(current_labels == 0)
            current_unlabeled = np.sum(current_labels == -1)
            
            avg_new_rn_score = unlabeled_scores[new_rn_mask].mean() if len(new_rn_indices) > 0 else 0
            
            iteration_info = {
                'iteration': iteration + 1,
                'threshold': current_threshold,
                'new_rns_added': len(new_rn_indices),
                'total_positives': current_positives,
                'total_rns': current_rns,
                'remaining_unlabeled': current_unlabeled,
                'avg_new_rn_score': avg_new_rn_score
            }
            self.rn_identification_history.append(iteration_info)
            
            print(f"Added {len(new_rn_indices)} new RNs")
            print(f"Current counts - P: {current_positives}, RN: {current_rns}, U: {current_unlabeled}")
            if len(new_rn_indices) > 0:
                print(f"New RNs average score: {avg_new_rn_score:.4f}")
            
            # Update threshold
            current_threshold = max(min_threshold, current_threshold - threshold_decay)
            if current_threshold <= min_threshold:
                print(f"Reached minimum threshold {min_threshold}")
                break
                
            if len(new_rn_indices) == 0:
                print("No new reliable negatives found")
                break
                
        print(f"\n--- RN Identification Complete ---")
        final_positives = np.sum(current_labels == 1)
        final_rns = np.sum(current_labels == 0)
        final_unlabeled = np.sum(current_labels == -1)
        
        print(f"Final counts:")
        print(f"  Positives: {final_positives}")
        print(f"  Reliable Negatives: {final_rns} (added {final_rns - initial_rns})")
        print(f"  Remaining Unlabeled: {final_unlabeled}")
        print(f"  RN Identification Rate: {((final_rns - initial_rns) / initial_unlabeled * 100):.1f}%")
        
        return current_labels
    
    def train_final_classifier(self, X, final_labels, early_stopping_rounds=10):
        """
        Phase 2: Train final classifier on ALL identified P + RN with specificity-based evaluation
        Since we use cross-validation separately, we train on all available labeled data here.
        """
        print("\n" + "="*70)
        print("PHASE 2: TRAINING FINAL CLASSIFIER (XGBoost with Specificity)")
        print("="*70)
        
        # Create training data - use ALL labeled data (P + RN)
        train_mask = final_labels != -1
        X_train = X[train_mask]
        y_train = final_labels[train_mask]
        
        train_positives = np.sum(y_train == 1)
        train_negatives = np.sum(y_train == 0)
        
        print(f"Final training set (using ALL labeled data):")
        print(f"  Positives: {train_positives}")
        print(f"  Negatives: {train_negatives}")
        print(f"  Total training samples: {len(y_train)}")
        print(f"  Positive ratio: {train_positives / len(y_train) * 100:.1f}%")
        
        # Train on all labeled data without validation split
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # Uncomment if you have the custom loss function
        # custom_loss = asymmetric_unified_focal_gev1(**self.loss_params_phase2)
        
        self.model = xgb.train(
            params=self.xgb_params_final,
            dtrain=dtrain,
            num_boost_round=self.xgb_params_final.get('n_estimators', 50),
            #obj=custom_loss,
            evals=[(dtrain, 'train')],
            #eval=specificity_eval,  
            verbose_eval=False,
            #maximize=True  
        )
        return self.model
    
    def predict(self, X):
        """Predict probabilities for new data"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        dtest = xgb.DMatrix(X)
        probs = self.model.predict(dtest)
        return probs
    
    def cross_validate(self, X, final_labels, cv_folds=5):
        """
        Perform cross-validation on labeled data (P + RN) with specificity tracking
        """
        print(f"\n--- Cross-Validation ({cv_folds}-fold) with Specificity ---")
        
        # Get labeled data only
        labeled_mask = final_labels != -1
        X_labeled = X[labeled_mask]
        y_labeled = final_labels[labeled_mask]
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=34)
        cv_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_labeled, y_labeled)):
            print(f"\nFold {fold + 1}/{cv_folds}")
            
            X_train_fold = X_labeled[train_idx]
            y_train_fold = y_labeled[train_idx]
            X_val_fold = X_labeled[val_idx]
            y_val_fold = y_labeled[val_idx]
            
            # Train model
            dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
            dval = xgb.DMatrix(X_val_fold, label=y_val_fold)
            
            # Uncomment if you have the custom loss function
            # custom_loss = asymmetric_unified_focal_gev1(**self.loss_params_phase2)
            
            fold_model = xgb.train(
                params=self.xgb_params_final,
                dtrain=dtrain,
                num_boost_round=self.xgb_params_final.get('n_estimators', 50),
                #obj=custom_loss,
                evals=[(dval, 'validation')],
                #feval=specificity_eval, 
                early_stopping_rounds=20,
                verbose_eval=True,
                #maximize=True
            )
            
            y_pred_probs = fold_model.predict(dval)
            print("Fold raw predictions (first 10):", y_pred_probs[:10])
            y_pred = (y_pred_probs >= 0.5).astype(int)
            
            fold_metrics = self.calculate_metrics(y_val_fold, y_pred, y_pred_probs)
            fold_metrics['fold'] = fold + 1
            cv_results.append(fold_metrics)
            
            print(f"  Accuracy: {fold_metrics['accuracy']:.4f}")
            print(f"  Precision: {fold_metrics['precision']:.4f}")
            print(f"  Recall: {fold_metrics['recall']:.4f}")
            print(f"  F1: {fold_metrics['f1']:.4f}")
            print(f"  AUC: {fold_metrics['auc']:.4f}")
            print(f"  AP: {fold_metrics['average_precision']:.4f}")
            print(f"  Specificity: {fold_metrics['specificity']:.4f}")  # Added specificity output
        
        # Average results
        avg_results = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'average_precision', 'specificity']:
            avg_results[f'avg_{metric}'] = np.mean([r[metric] for r in cv_results])
            avg_results[f'std_{metric}'] = np.std([r[metric] for r in cv_results])
        
        print(f"\n--- Cross-Validation Summary ---")
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'average_precision', 'specificity']:
            print(f"{metric.capitalize()}: {avg_results[f'avg_{metric}']:.4f} ± {avg_results[f'std_{metric}']:.4f}")
        
        return cv_results, avg_results
    
    def calculate_metrics(self, y_true, y_pred, y_prob):
        """Calculate comprehensive metrics including specificity"""
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        if len(np.unique(y_true)) > 1:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
            metrics['average_precision'] = average_precision_score(y_true, y_prob)
        else:
            metrics['auc'] = np.nan
            metrics['average_precision'] = np.nan
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics['tn'] = tn
        metrics['fp'] = fp  
        metrics['fn'] = fn
        metrics['tp'] = tp
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return metrics

def run_xgboost_pu_learning(X, initial_labels, gene_symbols=None):
    """
    Complete XGBoost PU Learning Pipeline
    
    Args:
        X: feature matrix (numpy array or pandas DataFrame)
        initial_labels: labels where 1=positive, 0=initial RNs, -1=unlabeled
        gene_symbols: optional list of gene symbols for results
    """
    
    if hasattr(X, 'values'):
        X = X.values
    
    learner = XGBoostPULearner()
    
    final_labels = learner.iterative_rn_identification(
        X, initial_labels,
        n_iterations=5,
        initial_threshold=0.90,
        threshold_decay=0.025,
        min_threshold=0.70
    )
    #
    labeled_mask = final_labels != -1
    X_labeled = X[labeled_mask]
    y_labeled = final_labels[labeled_mask]
    
    cv_results, avg_cv_results = learner.cross_validate(X_labeled, y_labeled, cv_folds=5)
    
    final_model = learner.train_final_classifier(X_labeled, y_labeled)
    #
    train_mask = final_labels != -1
    X_train = X[train_mask]
    y_train = final_labels[train_mask]
        
    train_positives = np.sum(y_train == 1)
    train_negatives = np.sum(y_train == 0)        
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    # Predict on training data
    train_preds = final_model.predict(dtrain)
    threshold = 0.5
    train_preds_binary = (train_preds >= threshold).astype(int)

    print("Train Accuracy:", accuracy_score(y_train, train_preds_binary))
    print("Train Precision:", precision_score(y_train, train_preds_binary))
    print("Train Recall:", recall_score(y_train, train_preds_binary))
    print("Train F1 Score:", f1_score(y_train, train_preds_binary))
    import matplotlib.pyplot as plt

    plt.hist(train_preds, bins=50, alpha=0.7)
    plt.title("Predicted probabilities on training data")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.show()

#
    unlabeled_mask = final_labels == -1
    X_unlabeled = X[unlabeled_mask]
    dtest = xgb.DMatrix(X_unlabeled)
    all_predictions = final_model.predict(dtest)
    #all_predictions = torch.sigmoid(torch.tensor(raw_scores)).numpy()
    #all_predictions = final_model.predict(X_unlabeled)
    print("Raw model predictions (before any transformation):", all_predictions[:10])
        
    results_data = []
    unlabeled_indices = np.where(unlabeled_mask)[0]
    
    for i, idx in enumerate(unlabeled_indices):
        pred_prob = all_predictions[i]
        if pred_prob > 0.8:
            confidence = 'High'
        elif pred_prob > 0.6:
            confidence = 'Medium'
        elif pred_prob >= 0.4:
            confidence = 'Low'
        elif pred_prob >= 0.2:
            confidence = 'Medium'
        else:
            confidence = 'High'
        
        result = {
            'NodeIndex': idx,
            'Prediction_Probability': pred_prob,
            'Confidence_Level': confidence
        }
        
        if gene_symbols is not None:
            result['GeneSymbol'] = gene_symbols[idx]
            
        results_data.append(result)
    
    results_df = pd.DataFrame(results_data)
    
    # Print summary
    print(f"\n{'='*70}")
    print("XGBOOST PU LEARNING COMPLETE")
    print(f"{'='*70}\n")
    
    print("Final Model Performance (CV Average):")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'average_precision']:
        print(f"  {metric.capitalize()}: {avg_cv_results[f'avg_{metric}']:.4f} ± {avg_cv_results[f'std_{metric}']:.4f}")
    
    print("\nPredictions on Unlabeled Data:")
    high_conf_pos = np.sum(all_predictions > 0.8)
    med_conf_pos = np.sum((all_predictions > 0.6) & (all_predictions <= 0.8))
    uncertain = np.sum((all_predictions >= 0.4) & (all_predictions <= 0.6))
    med_conf_neg = np.sum((all_predictions >= 0.2) & (all_predictions < 0.4))
    high_conf_neg = np.sum(all_predictions < 0.2)
    
    total_unlabeled = len(all_predictions)
    print(f"  High confidence positive (>0.8): {high_conf_pos} ({high_conf_pos/total_unlabeled*100:.1f}%)")
    print(f"  Medium confidence positive (0.6-0.8): {med_conf_pos} ({med_conf_pos/total_unlabeled*100:.1f}%)")
    print(f"  Uncertain (0.4-0.6): {uncertain} ({uncertain/total_unlabeled*100:.1f}%)")
    print(f"  Medium confidence negative (0.2-0.4): {med_conf_neg} ({med_conf_neg/total_unlabeled*100:.1f}%)")
    print(f"  High confidence negative (<0.2): {high_conf_neg} ({high_conf_neg/total_unlabeled*100:.1f}%)")
    
    return {
        'learner': learner,
        'final_model': final_model,
        'final_labels': final_labels,
        'results_df': results_df,
        'cv_results': cv_results,
        'avg_cv_results': avg_cv_results,
        'all_predictions': all_predictions
    }


X = embeddings.cpu().numpy()  
initial_labels = labels.squeeze().cpu().numpy()  
gene_symbols = nodes_df['GeneSymbol'].values  

results = run_xgboost_pu_learning(X, initial_labels, gene_symbols)
results_df = results['results_df']
print(results_df.head(10))
# results_df.
results_df = results_df.groupby(["NodeIndex", "GeneSymbol"]).first().reset_index()
results_df.to_excel('/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Results/DGI/XGBoost_combined.xlsx', index=False)