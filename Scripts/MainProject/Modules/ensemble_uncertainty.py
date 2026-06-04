import pandas as pd
import numpy as np
from pathlib import Path
from functools import reduce

import pandas as pd
import numpy as np
from pathlib import Path

def compute_ensemble_uncertainty(prediction_paths):
    model_probs = []
    model_preds = []
    all_node_ids = None
    model_names = []

    mc_epistemics = []
    bnn_epistemics = []

    support_columns = {}  

    for path in prediction_paths:
        model_name = Path(path).parent.parent.name.lower()
        clf_name = Path(path).parent.name.lower()
        full_name = f"{model_name}_{clf_name}"
        model_names.append(full_name)

        df = pd.read_csv(path)
        if all_node_ids is None:
            all_node_ids = df['node_index'].values

        probs = df['prob_positive'].values
        preds = (probs > 0.5).astype(int)

        model_probs.append(probs)
        model_preds.append(preds)

        # Track binary support
        support_columns[full_name] = preds

        # Track epistemic for MC Dropout and BNN
        if model_name == 'mc_dropout' and 'epistemic_uncertainty' in df.columns:
            mc_epistemics.append(df['epistemic_uncertainty'].values)
        if model_name == 'bnn' and 'epistemic_uncertainty' in df.columns:
            bnn_epistemics.append(df['epistemic_uncertainty'].values)

    print(f"Total number of models in ensemble: {len(model_names)}")

    all_probs = np.stack(model_probs)
    all_preds = np.stack(model_preds)

    prob_mean = all_probs.mean(axis=0)
    prob_std = all_probs.std(axis=0)
    label_majority = (np.sum(all_preds, axis=0) > (all_preds.shape[0] // 2)).astype(int)
    num_models_positive = np.sum(all_preds, axis=0)

    aleatoric = -np.mean(
        all_probs * np.log(all_probs + 1e-8) +
        (1 - all_probs) * np.log(1 - all_probs + 1e-8),
        axis=0
    )
    total = -prob_mean * np.log(prob_mean + 1e-8) - (1 - prob_mean) * np.log(1 - prob_mean + 1e-8)
    epistemic = total - aleatoric

    df_out = pd.DataFrame({
        'node_index': all_node_ids,
        'mean_probability': prob_mean,
        'prob_positive_std': prob_std,
        'aleatoric_uncertainty': aleatoric,
        'epistemic_uncertainty': epistemic,
        'total_uncertainty': total,
        'predicted_label': label_majority,
        'num_models_positive': num_models_positive,
        'support_count': num_models_positive  
    })

    for i, name in enumerate(model_names):
        df_out[f'prob_{name}'] = all_probs[i]
        df_out[f'label_{name}'] = all_preds[i]
        df_out[f'support_{name}'] = all_preds[i]  

    if mc_epistemics:
        mc_epistemics = np.stack(mc_epistemics)
        df_out['epistemic_mc_avg'] = mc_epistemics.mean(axis=0)

    if bnn_epistemics:
        bnn_epistemics = np.stack(bnn_epistemics)
        df_out['epistemic_bnn_avg'] = bnn_epistemics.mean(axis=0)

    return df_out