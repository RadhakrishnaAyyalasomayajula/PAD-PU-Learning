import numpy as np
from pathlib import Path

def load_embeddings(method="DGI", results_dir=None):
    if results_dir is None:
        # Automatically resolve to project root / Results
        results_dir = Path(__file__).resolve().parents[3] / "Results"
    else:
        results_dir = Path(results_dir)

    method_dir = results_dir / method.upper()
    emb_path = method_dir / f"{method.upper()}_embeddings.npy"
    std_path = method_dir / f"{method.upper()}_embeddings_std.npy"

    if not emb_path.exists() or not std_path.exists():
        raise FileNotFoundError(f"Embeddings not found for method: {method} at {emb_path}")

    mean_emb = np.load(emb_path)
    std_emb = np.load(std_path)

    return mean_emb, std_emb