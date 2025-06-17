#!/mnt/hcleroy/anaconda3/bin/python3

"""
embed_and_cluster.py — Build delay embeddings and apply k-means clustering.

Usage:
    ./scripts/03_embed_and_cluster.py \
        --input data/copepods/interim/phases.parquet \
        --output-dir data/copepods/processed \
        --columns speed,curvature_angle,torsion_angle \
        --K 30 \
        --n-clusters 30 \
        --tau 1,2,5,10 \
        --groupby label
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import json
import pickle

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.io import load_dataframe
from src.embedding import Embedding
from src.embedding_position import EmbeddingPosition


def parse_args():
    parser = argparse.ArgumentParser(description="Delay embedding and k-means clustering")
    parser.add_argument("--input", type=str, required=True, help="Path to phase-annotated input file (.csv or .parquet)")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save embedding and clustering outputs")
    parser.add_argument("--columns", type=str, required=True, help="Comma-separated feature columns (e.g. speed,curvature_angle)")
    parser.add_argument("--columns-trans",type=str, default=None,help="output directory for the embedding instance")
    parser.add_argument("--K", type=int, required=True, help="Delay length")
    #parser.add_argument("--tau", type=int, required=True, help="time of the Markov chain")
    parser.add_argument("--tau-values", type=str, required=True,
                        help="Comma-separated list of Tau values (e.g. 1,3,5,10)")
    parser.add_argument("--n-clusters", type=int, required=True, help="Number of k-means clusters")
    parser.add_argument("--random-state", type=int, default=0, help="Random seed for k-means")
    parser.add_argument("--min-length", type=int, default=20, help="Minimum trajectory length to retain")
    parser.add_argument("--groupby",type=str, default="ID",help="Name of the individual trajectories")
    parser.add_argument("--out-summary",type=str, default="markov_summary.json",help="output directory for the summary")
    parser.add_argument("--out-embedding",type=str, default="embedding.pkl",help="output directory for the embedding instance")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tau_values = list(map(int, args.tau_values.split(",")))

    # Load the phase data
    df = pd.read_parquet(args.input)


    # Build embedding object
    feature_cols = args.columns.split(",")
    if args.columns_trans is None:
        emb = Embedding(df, columns=feature_cols, ID_NAME=args.groupby)
    else :
        feature_cols_trans = args.columns_trans.split(",")
        emb = EmbeddingPosition(df, columns=feature_cols,columns_translated = feature_cols_trans,ID_NAME=args.groupby)

    # Construct delay embedding
    emb_matrix, flat_matrix = emb.make_embedding(args.K)
    print(f"[INFO] Embedded shape: {emb_matrix.shape}")

    # Cluster delay vectors
    labels = emb.make_cluster(n_clusters=args.n_clusters, random_state=args.random_state)
    print(f"[INFO] Clustered into {args.n_clusters} states")

    # Transition matrix    
    print(f"[INFO] transition matrix built")
    eig_val_10 = {}
    entropy_rates = {}
    timescales = {}

    for tau in tau_values:
        P = emb.make_transition_matrix(tau=tau)
        eig_val, eig_vec = np.linalg.eig(P)
        real_spectrum = np.real(eig_val)
        real_spectrum = real_spectrum[np.argsort(real_spectrum)][::-1]
        eig_val_10[tau] = real_spectrum[:10].tolist()

        h = emb.entropy_rate(P, emb.pi)
        ts = emb.implied_timescales(P, lag=tau)

        entropy_rates[tau] = h
        timescales[tau] = ts.tolist()

    result = {
        "entropy_rate": entropy_rates,
        "implied_timescales": timescales,
        "spectrum_tau": eig_val_10
    }

    with open(out_dir / args.out_summary , "w") as f:
        json.dump(result, f, indent=2)
    
    print(set(emb.labels).__len__())
    with open(out_dir / args.out_embedding, "wb") as f:
        pickle.dump(emb,f,protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[INFO] Transition matrix P: {P.shape}")
    print(f"[INFO] Stationary distribution π: {emb.pi.shape}")
    print(f"[INFO] Entropy rate: {h:.4f} bits")
    print(f"[INFO] Implied timescales: {ts[:5]}")
    print(f"[INFO] Saved to {out_dir}")


if __name__ == "__main__":
    main()
