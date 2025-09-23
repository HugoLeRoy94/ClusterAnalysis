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
from src.markov_analysis import *


def parse_args():
    parser = argparse.ArgumentParser(description="Scan entropy production over (K, n_clusters)")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the directory containing the input file.")
    parser.add_argument("--input-name", type=str, required=True, help="Name of the input file (without extension).")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the output file.")
    parser.add_argument("--extension", type=str, default="parquet", help="Extension of the input file (default: parquet).")
    parser.add_argument("--columns", type=str, default = None)
    parser.add_argument("--columns-trans",type=str, default=None,help="output directory for the embedding instance")
    parser.add_argument("--K-values", type=str, required=True,
                        help="Comma-separated list of K values (e.g. 5,7,10)")
    parser.add_argument("--n-clusters-values", type=str, required=True,
                        help="Comma-separated list of cluster numbers (e.g. 4,6,8)")
    parser.add_argument("--tau", type=int, required=True)
    parser.add_argument("--groupby", type=str, default="label")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-trajectories", type=int,default=None)
    parser.add_argument("--n-windows", type=int,default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    print(args.input_name)
    # Construct input and output file paths
    input_file = Path(args.input_path) / f"{args.input_name}.{args.extension}"
    output_file = Path(args.output_path) / f"entropy_scan_{args.input_name}.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(input_file)
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
            emb = Embedding(df, columns=feature_cols, ID_NAME=args.groupby,n_trajectories=args.n_trajectories,n_windows=args.n_windows)
        else :
            feature_cols_trans = args.columns_trans.split(",")
            emb = EmbeddingPosition(df, columns=feature_cols,columns_translated = feature_cols_trans,ID_NAME=args.groupby,n_trajectories = args.n_trajectories,n_windows=args.n_windows)
        
        emb_matrix, flat_matrix = emb.make_embedding(K)
        L = emb_matrix.shape[1]
        N_traj = emb_matrix.shape[0]

        for n_clusters in n_clusters_values:
            print(f"[INFO]   Clustering with {n_clusters} clusters")
            emb.make_cluster(n_clusters=n_clusters, random_state=args.random_state)
            #emb.make_transition_matrix(tau = args.tau)
            mkv = Markov(emb,tau = args.tau)
            h = mkv.compute_entropy_rate()

            results.append({
                "K": K,
                "n_clusters": n_clusters,
                "entropy_rate": h
            })

    df_result = pd.DataFrame(results)
    df_result.to_csv(output_file, index=False)
    print(f"[INFO] Results saved to {output_file}")


if __name__ == "__main__":
    main()