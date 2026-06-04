import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import f1_score
from Modules.classifiers import get_model, predict_mc_dropout, predict_bnn
from Modules.rn_embedding_plotter import plot_rn_embedding_distribution

def run_pu_refinement(embeddings, labels, classifier_name, save_dir, max_iter=10,
                      min_new_rns=5, uncertainty_threshold=0.05, prob_threshold=0.2):

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    refined_labels_dir = save_dir / "RefinedLabels"
    final_pred_dir = save_dir / "FinalPredictions"
    metrics_log_dir = save_dir / "MetricsLogs"
    for d in [refined_labels_dir, final_pred_dir, metrics_log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    labels = labels.copy()
    original_labels = labels.copy()
    pos_mask = labels == 1
    rn_mask = labels == 0
    unlab_mask = ~pos_mask & ~rn_mask

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)

    f1_history = []
    prev_f1 = 0

    for i in range(max_iter):
        train_mask = pos_mask | rn_mask
        X_train = embeddings[train_mask]
        y_train = labels[train_mask]

        if classifier_name in ['mc_dropout', 'bnn']:
            model = get_model(classifier_name, input_dim=embeddings.shape[1]).to(device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            model.train()
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
            for epoch in range(100):
                optimizer.zero_grad()
                output = model(X_train_tensor)
                loss = criterion(output, y_train_tensor)
                loss.backward()
                optimizer.step()
        else:
            model = get_model(classifier_name, X_train=X_train, y_train=y_train)

        unlab_indices = np.where(unlab_mask)[0]
        X_unlab = embeddings[unlab_indices]
        X_unlab_tensor = X_tensor[unlab_indices]

        adjusted_prob_threshold = prob_threshold
        adjusted_uncertainty_threshold = uncertainty_threshold
        if classifier_name == 'random_forest':
            adjusted_prob_threshold = 0.1

        if classifier_name == 'mc_dropout':
            samples = predict_mc_dropout(model, X_unlab_tensor, n_samples=30)
        elif classifier_name == 'bnn':
            samples = predict_bnn(model, X_unlab_tensor, n_samples=30)

        if classifier_name in ['mc_dropout', 'bnn']:
            mean_probs = samples.mean(axis=0)
            std_probs = samples.std(axis=0)
            entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=1)
            epistemic_uncertainty = entropy
            mean_pos_probs = mean_probs[:, 1]
            new_rn_mask = (mean_pos_probs < adjusted_prob_threshold) & (epistemic_uncertainty < adjusted_uncertainty_threshold)
        else:
            if hasattr(model, 'calibrated_classifiers_'):
                probs_all = [clf.predict_proba(X_unlab)[:, 1] for clf in model.calibrated_classifiers_]
                probs_array = np.stack(probs_all)
                probs = probs_array.mean(axis=0)
            else:
                probs = model.predict_proba(X_unlab)[:, 1]
            mean_pos_probs = probs
            entropy = -probs * np.log(probs + 1e-8) - (1 - probs) * np.log(1 - probs + 1e-8)
            new_rn_mask = probs < adjusted_prob_threshold

        new_rn_indices = unlab_indices[new_rn_mask]
        print(f"[{classifier_name.upper()}] Iter {i+1}: Selected {len(new_rn_indices)} new RNs")

        labels[new_rn_indices] = 0
        rn_mask = labels == 0
        unlab_mask = ~pos_mask & ~rn_mask

        y_eval = original_labels[train_mask]
        if np.any(y_eval == 1):
            if classifier_name in ['mc_dropout', 'bnn']:
                eval_samples = predict_mc_dropout(model, X_tensor[train_mask], n_samples=30) \
                    if classifier_name == 'mc_dropout' else predict_bnn(model, X_tensor[train_mask], n_samples=30)
                y_pred = (eval_samples.mean(axis=0)[:, 1] > 0.5).astype(int)
            else:
                y_pred = model.predict(X_train)
            mask = (y_eval == 0) | (y_eval == 1)
            f1 = f1_score(y_eval[mask], y_pred[mask], average='binary', pos_label=1)
            delta_f = f1 - prev_f1
            tp = np.sum((y_eval == 1) & (y_pred == 1))
            fn = np.sum((y_eval == 1) & (y_pred == 0))
            fnr = fn / (tp + fn + 1e-8)
            if len(new_rn_indices) < min_new_rns or delta_f < 0.001 or fnr > 0.05:
                break
            prev_f1 = f1
            f1_history.append((i + 1, f1))

    np.save(refined_labels_dir / "refined_labels.npy", labels)

    try:
        metadata_path = "/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/CuratedData/LargestComponents/Nodes_with_properties_FC.csv"
        meta_df = pd.read_csv(metadata_path)[['NodeIndex', 'GeneSymbol']].rename(columns={'NodeIndex': 'node_index'})
    except Exception as e:
        print(f"⚠️ Could not load metadata CSV: {e}")
        meta_df = pd.DataFrame()

    label_types = []
    for i, lbl in enumerate(labels):
        if lbl == 1:
            label_types.append("positive")
        elif lbl == 0:
            if original_labels[i] == 0:
                label_types.append("RN")
            elif original_labels[i] == -1:
                label_types.append("inferred_negative")
            else:
                label_types.append("RN_or_inferred")
        else:
            label_types.append("unlabeled")

    label_df = pd.DataFrame({
        'node_index': np.arange(len(labels)),
        'final_label': labels,
        'label_type': label_types
    })

    if not meta_df.empty:
        label_df = label_df.merge(meta_df, on='node_index', how='left')
    label_df.to_csv(refined_labels_dir / "final_labels_with_index.csv", index=False)

    if classifier_name in ['mc_dropout', 'bnn']:
        final_samples = predict_mc_dropout(model, X_tensor, n_samples=30) \
            if classifier_name == 'mc_dropout' else predict_bnn(model, X_tensor, n_samples=30)
        final_mean_probs = final_samples.mean(axis=0)[:, 1]
        final_entropy = -np.sum(final_samples.mean(axis=0) * np.log(final_samples.mean(axis=0) + 1e-8), axis=1)
        epistemic_uncertainty = final_entropy
        np.save(final_pred_dir / "posterior_samples.npy", final_samples)

    else:
        if hasattr(model, 'calibrated_classifiers_'):
            probs_all = [clf.predict_proba(embeddings)[:, 1] for clf in model.calibrated_classifiers_]
            probs_array = np.stack(probs_all)
            final_mean_probs = probs_array.mean(axis=0)
        else:
            final_mean_probs = model.predict_proba(embeddings)[:, 1]
        epistemic_uncertainty = -final_mean_probs * np.log(final_mean_probs + 1e-8) - \
                                 (1 - final_mean_probs) * np.log(1 - final_mean_probs + 1e-8)

    pred_df = pd.DataFrame({
        'node_index': np.arange(len(final_mean_probs)),
        'prob_positive': final_mean_probs,
        'epistemic_uncertainty': epistemic_uncertainty,
        'entropy': epistemic_uncertainty,
        'predicted_label': (final_mean_probs > 0.5).astype(int)
    })
    if not meta_df.empty:
        pred_df = pred_df.merge(meta_df, on='node_index', how='left')
    pred_df.to_csv(final_pred_dir / "final_predictions.csv", index=False)

    pd.DataFrame(f1_history, columns=["iteration", "f1"]).to_csv(metrics_log_dir / "f1_history.csv", index=False)

    try:
        pred_labels_tensor = torch.tensor(pred_df['predicted_label'].values)
        rn_indices_final = np.where(pred_df['predicted_label'].values == 0)[0]
        tsne_path = final_pred_dir / "tsne_final_predictions.png"

        plot_rn_embedding_distribution(
            embeddings=embeddings,
            labels_tensor=pred_labels_tensor,
            rn_indices=rn_indices_final,
            method="TSNE",
            save_path=str(tsne_path)
        )
        print(f"✅ Saved final t-SNE plot to: {tsne_path}")
    except Exception as e:
        print(f"⚠️ Failed to generate final t-SNE plot: {e}")