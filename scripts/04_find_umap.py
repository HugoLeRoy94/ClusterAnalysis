#!/mnt/hcleroy/anaconda3/bin/python3

"""
05_find_umap.py — Compute 2D UMAP of the flatten embedding matrix.
Optionally overlay cluster centers and subsample data to reduce load.

Usage:
    ./scripts/04_find_umap.py \
        --embedding data/copepods/processed/ \
        --output data/copepods/processed/umap.npy \
        --subsample 1000 \
        --cluster-centers True \
        --n-neighbors 100 \
        --min-dist 0.1
"""

import argparse
import numpy as np
from pathlib import Path
import sys
import pickle

# Add src/ to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from umap import UMAP


def parse_args():
    parser = argparse.ArgumentParser(description="Compute 2D UMAP from embedding matrix")
    parser.add_argument("--input", type=str, required=True, help="Path to embedding.pkl and markov.pkl")
    parser.add_argument("--output", type=str, required=True, help="Path to save UMAP coordinates (.npy)")
    parser.add_argument("--subsample", type=int, default=None, help="Number of rows to subsample from the flattened matrix")
    parser.add_argument("--cluster-centers", type=bool, default=False, help="Optional bool whether we plot the clusters")
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP: number of neighbors")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP: minimum distance")
    parser.add_argument("--random-state", type=int, default=None, help="UMAP: random seed")
    parser.add_argument("--n-components", type=int, default=2,help="UMAP: number of umap components")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and flatten embedding    
    with open(args.input+"embedding.pkl", "rb") as f:
        emb = pickle.load(f)
    #with open(args.input+"markov.pkl","rb") as f:
    #    mkv = pickle.load(f)
    print(f"[INFO] Loaded embedding")
    print(emb)     

    # Subsample if needed
    if args.subsample is not None and args.subsample < emb.flatten_embedding_matrix.shape[0]:
        rng = np.random.default_rng(args.random_state)
        indices = rng.choice(emb.flatten_embedding_matrix.shape[0], size=args.subsample, replace=False)
        data = emb.flatten_embedding_matrix[indices]
        print(f"[INFO] Subsampled {data.shape[0]} rows from flattened matrix")
    else:
        data = emb.flatten_embedding_matrix
        indices = None
    
    combined =np.array([])
    if args.cluster_centers:
        # Concatenate data and centers before UMAP
        combined = np.append(data, emb.cluster_centers_, axis=0)
    else :
        combined = data
    reducer = UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=args.n_components,
        metric="euclidean",
    )
    reduced_all = reducer.fit_transform(combined)
    print(f"[INFO] UMAP computed → shape {reduced_all.shape}")

    if args.cluster_centers:
        # Separate out points and cluster centers
        N = data.shape[0]
        reduced_points = reduced_all[:N]
        reduced_centers = reduced_all[N:N + emb.cluster_centers_.shape[0]]
        print(reduced_centers.shape)
        print(reduced_points.shape)
        print(f"[INFO] Saved UMAP coordinates to {output_dir}/umap_centers.npy")
        np.save(output_dir / "umap_centers.npy", reduced_centers)
    else:
        reduced_points = reduced_all

    np.save(output_dir / "umap_points.npy", reduced_points)
    print(f"[INFO] Saved UMAP coordinates to {args.output}")

    # Optional: save subsampled labels
    if indices is not None:
        np.save(output_dir / "umap_indices.npy", indices)
        print(f"[INFO] Saved subsampled labels to {output_dir / 'umap_indices.npy'}")


if __name__ == "__main__":
    main()
