import numpy as np
from Modules.label_handler import *
from Modules.rn_uncertainty import *
import pandas as pd 
import os 

def select_confident_rns(rn_sets, embeddings, labels, embedding_probs,
                         top_k_ratio=1.0, k_distance=5,
                         weight_scheme="uniform", max_final_rns=None,
                         export_base=None):
    """
    Selects the most confident Reliable Negatives (RNs) by combining multiple uncertainty measures.

    Args:
        rn_sets (List[np.ndarray]): List of RN index sets from different strategies.
        embeddings (np.ndarray): Node embeddings of shape (N, D).
        embedding_probs (np.ndarray): Probabilistic representation per node (N, D) for entropy calc.
        top_k_ratio (float): Ratio of number of confident RNs to number of known positives.
        k_distance (int): K for k-NN distance in distance-based uncertainty.
        weight_scheme (str): "uniform" or "custom" weights for each score.
        max_final_rns (int, optional): Maximum number of RNs to select.

    Returns:
        final_rns (np.ndarray): Indices of selected reliable negatives.
        rn_scores (dict): Score breakdown per RN.
        all_rn_candidates (np.ndarray): All RN indices considered.
        combined_score (np.ndarray): Final score per RN used for selection.
    """
    # Aggregate candidate RN indices (union of all sets)
    all_rn_candidates = np.array(sorted(set(np.concatenate(rn_sets))), dtype=int)

    # --- Score 1: Agreement Score (RAW + Normalized) ---
    agreement_counts = agreement_score(rn_sets)
    agreement_scores_raw = np.array([agreement_counts.get(idx, 0) for idx in all_rn_candidates])

    # --- Score 2: Entropy ---
    if embedding_probs is None:
        raise ValueError("embedding_probs must be provided for entropy computation.")
    entropy_scores = entropy_based(embedding_probs[all_rn_candidates])

    # --- Score 3: Distance to positives ---
    distance_scores = distance_based_uncertainty(embeddings, all_rn_candidates, labels, k=k_distance)

    # --- Normalize all scores ---
    def normalize(x):
        x = np.asarray(x)
        return (x - x.min()) / (x.max() - x.min() + 1e-9)

    norm_agreement = normalize(agreement_scores_raw)
    norm_entropy = 1 - normalize(entropy_scores) 
    norm_distance = normalize(distance_scores)

    # --- Combine Scores ---
    if weight_scheme == "uniform":
        combined_score = (norm_agreement + norm_entropy + norm_distance) / 3
    elif weight_scheme == "custom":
        combined_score = (
            0.10 * norm_agreement +
            0.45 * norm_entropy +
            0.45 * norm_distance
        )
    else:
        raise NotImplementedError("Only 'uniform' and 'custom' weighting schemes are supported.")

    # --- Select top RNs ---
    top_k = int(len(combined_score) * top_k_ratio)
    sorted_idx = np.argsort(combined_score)[-top_k:]
    sorted_rns = all_rn_candidates[sorted_idx]

    # Apply maximum RN cap if provided
    if max_final_rns is not None and len(sorted_rns) > max_final_rns:
        sorted_rns = sorted_rns[:max_final_rns]
        sorted_idx = sorted_idx[:max_final_rns]

    final_rns = sorted_rns

    print(f"Selected {len(final_rns)} reliable negatives from {len(all_rn_candidates)} candidates.")

    rn_scores = {
        'all_rn_candidates': all_rn_candidates,
        'agreement': dict(zip(all_rn_candidates, norm_agreement)),
        'agreement_raw': dict(zip(all_rn_candidates, agreement_scores_raw)),
        'entropy': dict(zip(all_rn_candidates, norm_entropy)),
        'distance': dict(zip(all_rn_candidates, norm_distance)),
        'combined': dict(zip(all_rn_candidates, combined_score)),
    }

    if export_base is None:
        raise ValueError("export_base path must be provided to save RN outputs.")
    os.makedirs(export_base, exist_ok=True)

    score_csv_path = os.path.join(export_base, "Final_RNs.csv")
    rn_df = pd.DataFrame({
        'node_index': final_rns,
        'combined_score': combined_score[sorted_idx],
        'agreement_score': norm_agreement[sorted_idx],
        'entropy_score': norm_entropy[sorted_idx],
        'distance_score': norm_distance[sorted_idx],
        'agreement_raw': agreement_scores_raw[sorted_idx]
    })
    rn_df.to_csv(score_csv_path, index=False)
    print(f"✅ Saved RN score summary to: {score_csv_path}")

    # Save RN metadata with gene symbols
    try:
        metadata_path = "/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/CuratedData/LargestComponents/Nodes_with_properties_FC.csv"
        nodes_meta = pd.read_csv(metadata_path)
        meta_df = nodes_meta.loc[final_rns, ['NodeIndex', 'GeneSymbol']].copy()
        meta_df = meta_df.rename(columns={'NodeIndex': 'node_index'})
        full_rn_df = rn_df.merge(meta_df, on='node_index', how='left')
        full_rn_df = full_rn_df.sort_values(by='combined_score', ascending=False)

        meta_csv_path = os.path.join(export_base, "Final_RNs_with_GeneSymbols.csv")
        full_rn_df.to_csv(meta_csv_path, index=False)
        print(f"✅ Saved RN metadata with gene symbols to: {meta_csv_path}")

    except Exception as e:
        print(f"⚠️ Failed to save RN metadata with symbols: {e}")


    return final_rns, rn_scores, all_rn_candidates, combined_score
