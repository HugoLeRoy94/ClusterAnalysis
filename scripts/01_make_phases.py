#!/mnt/hcleroy/anaconda3/bin/python3

"""
make_phases.py — Compute speed, turning angle, and motion phases for copepod trajectories.

Usage:
    ./scripts/01_make_phases.py --input data/copepods/raw/copepods_R1_000rpm.csv \
                                --output data/copepods/interim/ \
                                --smooth True \
                                --window 50 \
                                --polyorder 3 \
                                --max-length 1000
                            
"""

import numpy as np
import argparse
from pathlib import Path
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.preprocessing import (compute_phases,split_trajectories,filter_trajectories,smooth_trajectory_savgol)
from src.io import load_dataframe, save_dataframe


def parse_args():
    parser = argparse.ArgumentParser(description="Compute motion phases from trajectory data.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV or Parquet trajectory file")
    parser.add_argument("--output", type=str, required=True, help="Path to save phase-annotated Parquet file")    
    parser.add_argument("--smooth",type=bool,default=False,help="if true smooth out the data before processing")    
    parser.add_argument("--groupby", type=str, default='label', help = "column names for each trajectory")
    parser.add_argument("--sortby", type=str, default='frame', help = "column names corresponding to time")
    parser.add_argument("--dt", type=float, default=1.0, help="Time step between samples")
    parser.add_argument("--cols", type=str, default="x,y,z", help="Comma-separated coordinate columns (default: x,y)")
    parser.add_argument("--window",type=int,default=7, help="window size of the smoothing")
    parser.add_argument("--polyorder",type=int,default=3,help="order of the polynomial for smoothing")
    parser.add_argument("--max-length",type=int,default=1000,help="maximum duration of the trajectories")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load raw data
    df = load_dataframe(args.input)
    cols = args.cols.split(",")
    print(f"[INFO] Loaded input with {len(df)} rows using columns {cols}")
    df = filter_trajectories(df,min_length=100,groupby=args.groupby)
    if args.smooth:
        df = smooth_trajectory_savgol(df,
                                    columns=cols,
                                    window=args.window,
                                    polyorder=args.polyorder,
                                    groupby=args.groupby)    
    # Compute kinematic features and phase
    df_phase = compute_phases(df, column_names=cols, dt=args.dt,groupby = args.groupby)
    print(f"[INFO] minimum length trajectory {df_phase.groupby(args.groupby).size().min()}")


    longest_traj = df_phase[df_phase['label'] == df_phase['label'].value_counts().idxmax()]
    save_dataframe(longest_traj,args.output+'longest_trajectory.csv')

    # extract fixed length trajectories
    
    df_phase = split_trajectories(df_phase,chunk_size=args.max_length,groupby=args.groupby,sort_values=args.sortby)
    #df_phase['torsion_angle'] = abs(df_phase['torsion_angle'])

    # Save
    save_dataframe(df_phase, args.output+'phases.parquet')
    print(f"[INFO] Saved phase-annotated data to: {args.output}/phases.parquet")


if __name__ == "__main__":
    main()
