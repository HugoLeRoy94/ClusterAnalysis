./../../scripts/01_make_phases.py \
        --input ../../data/toy_model/raw/helix_and_straight_lines.parquet \
        --output ../../data/toy_model/interim/ \
        --smooth True \
        --window 5 \
        --polyorder 3  \
        --max-length 997


./../../scripts/02_compute_entropy_production.py \
        --input ../../data/toy_model/interim/phases.parquet \
        --output-file ../../data/toy_model/interim/entropy_scan.csv \
        --columns speed,curvature_angle,torsion_angle \
        --columns-trans x,y,z \
        --K-values 1,3,5,10,20,30,40,50 \
        --n-clusters-values 2,3,4,5,10,15,20,30,40,50 \
        --tau 1

./../../scripts/03_embed_and_cluster.py \
        --input ../../data/toy_model/interim/phases.parquet \
        --output-dir ../../data/toy_model/processed \
        --columns speed,curvature_angle,torsion_angle \
        --columns-trans x,y,z \
        --K 30 \
        --n-clusters 30 \
        --tau 1,2,5,10,15,20,30,40,50,75,100,125,150,175,200 \
        --groupby label

./../../scripts/04_find_umap.py \
        --embedding ../../data/toy_model/processed/embedding.pkl \
        --output ../../data/toy_model/processed/umap.npy \
        --subsample 1000 \
        --cluster-centers True \
        --n-neighbors 100 \
        --min-dist 0.1