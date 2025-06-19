# structure of the project

<pre>
cluster-analysis/               ← project root (git repo)
├─ README.md                      ← project overview & quick-start
│
├─ src/                           ← importable Python package
│  ├─ __init__.py                 ← exports public API
│  ├─ io.py                       ← load/save DataFrames & npy files
│  ├─ preprocessing.py            ← trajectory preprocessing functions
│  ├─ embedding.py                ← Embedding class only
│  ├─ viz.py                      ← Matplotlib/seaborn helpers
│  └─ __pycache__/                ← auto-generated Python cache
│
├─ notebooks/                    ← analysis notebooks
│  ├─ copepods/                  
│  │  ├─ 01_trajectory_to_phase.ipynb
│  │  ├─ 02_optimize_entropy_production.ipynb
│  │  ├─ 03_markovian_analysis.ipynb
│  │  ├─ 04_UMAP.ipynb
│  │  ├─ 05_predict_trajectory.ipynb
│  │  └─ project_trajectory_on_phi2.ipynb
│  └─ toy_model/                 ← (currently empty)
│
├─ scripts/                      ← CLI scripts
│  ├─ 01_make_phases.py
│  ├─ 02_compute_entropy_production.py
│  ├─ 03_embed_and_cluster.py
│  └─ 04_find_umap.py
│
├─ unit_test/                    ← tests & debugging notebooks
│  └─ test_torsion_computation.ipynb
│
├─ reports/                      ← LaTeX report and figures
│  ├─ Intro.tex
│  ├─ report_1.tex
│  ├─ report_1.pdf
│  ├─ report_1.aux / .log / .out / .fls / .fdb_latexmk / .synctex.gz
│  └─ trajectory_length.pdf
│
└─ data/                         ← git-ignored data
   └─ copepods/
      ├─ raw/
      │  ├─ copepods_R1_000rpm.csv
      │  ├─ info.txt
      │  ├─ info_copepods_R1_000rpm.txt
      │  ├─ subsetCopepodsStillWater.7z
      │
      ├─ interim/
      │  ├─ cluster_centers.npy
      │  ├─ entropy_scan.csv      
      │  ├─ longest_trajectory.csv
      │  ├─ phases.parquet
      │
      └─ processed/
         ├─ embedding.pkl
         ├─ markov_summary.json
         ├─ umap_....npy

      └─ processed/                ← embeddings.pkl
</pre>

# How it works

scripts are meant to be executed in the terminal in the order given by the number. Each of the script produce a file either directly needed by the next script, or information useful to execute the next script. Notebooks with matching numbers are sanity checks of the output files. 

In order to use these script on the different dataset, the important parameters to adjust are :
- "--input" / "--output" : path of the input and output files
- "--groupby" name of the column of the data that labels individual trajectories
- "--sortby" name of the time/frame/etc... column
