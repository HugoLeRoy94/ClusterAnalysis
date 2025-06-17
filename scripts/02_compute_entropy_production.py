#!/mnt/hcleroy/anaconda3/bin/python3

"""
compute_entropy_production.py — Build delay embeddings, apply k-means, build the Markov model, and compute the entropy rate.

Usage:
    ./scripts/02_compute_entropy_production.py \
        --input data/copepods/interim/phases.parquet \
        --output-file data/copepods/interim/entropy_scan.csv \
        --columns speed,curvature_angle,torsion_angle \
        --K-values 1,3,5,10,20,30,40,50 \
        --n-clusters-values 2,3,4,5,10,15,20,30,40,50 \
        --tau 1
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.io import load_dataframe
from src.embedding import Embedding
from src.embedding_position import EmbeddingPosition


def parse_args():
    parser = argparse.ArgumentParser(description="Scan entropy production over (K, n_clusters)")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--columns", type=str, default = None)
    parser.add_argument("--columns-trans",type=str, default=None,help="output directory for the embedding instance")
    parser.add_argument("--K-values", type=str, required=True,
                        help="Comma-separated list of K values (e.g. 5,7,10)")
    parser.add_argument("--n-clusters-values", type=str, required=True,
                        help="Comma-separated list of cluster numbers (e.g. 4,6,8)")
    parser.add_argument("--tau", type=int, required=True)
    parser.add_argument("--groupby", type=str, default="label")
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    out_file = Path(args.output_file)
    #out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(args.input)
    if args.columns is not None:
        feature_cols = args.columns.split(",")
    else :
        feature_cols = []

    results = []
    K_values = list(map(int, args.K_values.split(",")))
    n_clusters_values = list(map(int, args.n_clusters_values.split(",")))


    for K in K_values:
        print(f"[INFO] Processing K = {K}")        
        #emb = Embedding(df, columns=feature_cols,ID_NAME='label')
        # Build embedding object
        if args.columns_trans is None:
            emb = Embedding(df, columns=feature_cols, ID_NAME=args.groupby)
        else :
            feature_cols_trans = args.columns_trans.split(",")
            emb = EmbeddingPosition(df, columns=feature_cols,columns_translated = feature_cols_trans,ID_NAME=args.groupby)
        
        emb_matrix, flat_matrix = emb.make_embedding(K)
        L = emb_matrix.shape[1]
        N_traj = emb_matrix.shape[0]        

        for n_clusters in n_clusters_values:
            print(f"[INFO]   Clustering with {n_clusters} clusters")
            emb.make_cluster(n_clusters=n_clusters, random_state=args.random_state)
            emb.make_transition_matrix(tau = args.tau)
            h = emb.entropy_rate(emb.P, emb.pi)

            results.append({
                "K": K,
                "n_clusters": n_clusters,
                "entropy_rate": h
            })

    df_result = pd.DataFrame(results)
    output_path = out_file
    df_result.to_csv(output_path, index=False)
    print(f"[INFO] Results saved to {output_path}")


if __name__ == "__main__":
    main()