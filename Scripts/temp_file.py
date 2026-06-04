import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

loaded_embeddings = np.load('avg_embeddings.npy')
embeddings_tensor = torch.tensor(loaded_embeddings, dtype=torch.float32)


class NeuralNet(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=1):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  
        return x
#
class nnPULoss(torch.nn.Module):
    def __init__(self, prior, gamma=1, beta=0, nnpu=True):
        super(nnPULoss, self).__init__()
        if not 0 < prior < 1:
            raise ValueError("The class prior should be in (0, 1)")
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.nnpu = nnpu

    def forward(self, x, t):
        positive = (t == 1)
        unlabeled = (t == -1)

        n_positive = max(1., positive.sum().item())  # To avoid division by zero
        n_unlabeled = max(1., unlabeled.sum().item())

        y_positive = F.sigmoid(-x)  # For positive samples
        y_unlabeled = F.sigmoid(x)  # For unlabeled samples (inverted risk)
        #
#        y_positive = F.binary_cross_entropy_with_logits(-x[positive], torch.ones_like(x[positive]), reduction='sum')
#        y_unlabeled = F.binary_cross_entropy_with_logits(x[unlabeled], torch.zeros_like(x[unlabeled]), reduction='sum')
        # Positive and unlabeled risk
        positive_risk = self.prior * y_positive[positive].sum() / n_positive
        unlabeled_risk = (y_unlabeled[unlabeled].sum() / n_unlabeled) - (
                    self.prior * y_unlabeled[positive].sum() / n_positive)

        objective = positive_risk + unlabeled_risk

        if self.nnpu:
            if unlabeled_risk < -self.beta:
                objective = -self.gamma * unlabeled_risk
                total_loss = objective
            else:
                total_loss = objective
        else:
            total_loss = objective
        return total_loss
#
def pu_loss(x, t, prior, nnpu=True):
    return nnPULoss(prior=prior, nnpu=nnpu)(x, t)

model = NeuralNet(input_dim=embedding_dim, hidden_dim=64, output_dim=1)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

prior_pos = 0.1
beta = 0
gamma = 1
nnpu = True

# Instantiate the PU loss function (assuming nnPULoss supports nnpu argument)
pu_loss_fn = nnPULoss(prior=prior_pos, nnpu=nnpu)

n_folds = 5
n_epochs = 100

kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

final_test_mask = test_mask.copy()  # Assuming test_mask is defined as a numpy array
results = []

for fold, (train_idx, spy_idx) in enumerate(kf.split(train_indices)):
    fold_train_mask = np.zeros(len(labels_df), dtype=bool)
    fold_test_mask = final_test_mask.copy()

    # Mark 80% known labels as training, 20% as spy (held-out)
    fold_train_mask[train_indices[train_idx]] = True
    fold_test_mask[train_indices[spy_idx]] = True
    fold_train_mask[train_indices[spy_idx]] = False

    best_val_loss = float('inf')
    counter = 0
    patience = 100

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(embeddings)  # embeddings: tensor of node embeddings, shape [N, emb_dim]

        if torch.isnan(out).any():
            print("Warning: output contains NaNs!")

        # Convert masks to torch tensors on the same device as model output
        device = out.device
        train_mask_tensor = torch.tensor(fold_train_mask, dtype=torch.bool, device=device)

        # Initialize labels tensor: -1 for unlabeled, set positives where known
        labels_tensor = torch.full_like(out, fill_value=-1, dtype=torch.float, device=device)
        labels_tensor[train_mask_tensor] = labels[train_mask_tensor]  # labels should be a tensor on same device

        loss = pu_loss_fn(out, labels_tensor)

        # Apply nnPU scaling directly to loss, NOT learning rate
        if loss < -beta:
            loss = -gamma * loss

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                train_out = model(embeddings)[train_mask_tensor]
                train_preds = torch.sigmoid(train_out).cpu().numpy()
                train_preds_binary = (train_preds >= 0.5).astype(int)

                spy_mask_tensor = torch.tensor(np.isin(np.arange(len(labels_df)), train_indices[spy_idx]), dtype=torch.bool, device=device)
                spy_out = model(embeddings)[spy_mask_tensor]
                spy_preds = torch.sigmoid(spy_out).cpu().numpy()
                spy_preds_binary = (spy_preds >= 0.5).astype(int)

                train_labels_np = labels[train_mask_tensor].cpu().numpy()
                spy_labels_np = labels[spy_mask_tensor].cpu().numpy()

                train_accuracy = accuracy_score(train_labels_np, train_preds_binary)
                spy_accuracy = accuracy_score(spy_labels_np, spy_preds_binary)

                true_positives = ((spy_preds_binary == 1) & (spy_labels_np == 1)).sum()
                false_negatives = ((spy_preds_binary == 0) & (spy_labels_np == 1)).sum()
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

                print(f'Epoch {epoch + 1}, Fold {fold}, Loss: {loss.item():.4f}, Train Acc: {train_accuracy:.4f}, Spy Recall: {recall:.4f}')

        # Early stopping based on validation loss
        if loss.item() < best_val_loss:
            best_val_loss = loss.item()
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} in fold {fold}")
            break

    # After training: eval on all
    model.eval()
    with torch.no_grad():
        all_out = model(embeddings)
        probabilities = torch.sigmoid(all_out).cpu().numpy()

        spy_mask_tensor = torch.tensor(np.isin(np.arange(len(labels_df)), train_indices[spy_idx]), dtype=torch.bool, device=device)
        spy_out = all_out[spy_mask_tensor]
        spy_preds = torch.sigmoid(spy_out).cpu().numpy()
        spy_preds_binary = (spy_preds >= 0.5).astype(int)

        train_mask_tensor = torch.tensor(fold_train_mask, dtype=torch.bool, device=device)
        train_labels_np = labels[train_mask_tensor].cpu().numpy()
        spy_labels_np = labels[spy_mask_tensor].cpu().numpy()

        train_out = all_out[train_mask_tensor]
        train_preds = torch.sigmoid(train_out).cpu().numpy()
        train_preds_binary = (train_preds >= 0.5).astype(int)

        train_accuracy = accuracy_score(train_labels_np, train_preds_binary)
        spy_accuracy = accuracy_score(spy_labels_np, spy_preds_binary)

        true_positives = ((spy_preds_binary == 1) & (spy_labels_np == 1)).sum()
        false_negatives = ((spy_preds_binary == 0) & (spy_labels_np == 1)).sum()
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

        # Save results for unknown/test nodes
        unknown_indices = test_indices  # Assuming test_indices is a list or array of test node indices
        unknown_probabilities = probabilities[unknown_indices]

        for idx, prob in zip(unknown_indices, unknown_probabilities):
            gene_symbol = nodes_df.loc[nodes_df['NodeIndex'] == idx, 'GeneSymbol'].values[0]
            results.append({
                "NodeIndex": idx,
                "GeneSymbol": gene_symbol,
                f"fold_{fold}_prob": float(prob),
                f"fold_{fold}_recall": recall
            })

import pandas as pd
df_all_results = pd.DataFrame(results)
df_pivoted = df_all_results.groupby(["NodeIndex", "GeneSymbol"]).first().reset_index()
df_pivoted.to_excel('/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Results/nnPU_results_combined.xlsx', index=False)
