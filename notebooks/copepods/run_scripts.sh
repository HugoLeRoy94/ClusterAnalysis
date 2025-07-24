../../scripts/01_make_phases.py \
        --input ../../data/copepods/raw/copepods_R1_000rpm.csv \
        --output ../../data/copepods/interim/ \
        --smooth True \
        --window 5 \
        --polyorder 3 \
        --max-length 997
                            

#--columns speed,curvature_angle,torsion_angle \

../../scripts/02_compute_entropy_production.py \
        --input ../../data/copepods/interim/phases.parquet \
        --output-file ../../data/copepods/interim/entropy_scan.csv \
        --columns-trans x,y,z \
        --K-values 3,5,10,20,30,40,50 \
        --n-clusters-values 2,3,4,5,10,15,20,30,40,50,100,250,500,750,1000,2000,3000,4000,5000 \
        --tau 1 \
        --n-trajectories 1000 \
        --n-windows 5000


../../scripts/03_embed_and_cluster.py \
        --input ../../data/copepods/interim/phases.parquet \
        --output-dir ../../data/copepods/processed \
        --columns-trans x,y,z \
        --K 50 \
        --n-clusters 2000 \
        --tau 1,2,3,5,7,10,25,50,100,200 \
        --groupby label \
        --n-trajectories 100

../../scripts/04_find_umap.py \
        --input ../../data/copepods/processed/ \
        --output ../..data/copepods/processed/umap.npy \
        --subsample 1000 \
        --cluster-centers True \
        --n-neighbors 100 \
        --min-dist 0.1                                    