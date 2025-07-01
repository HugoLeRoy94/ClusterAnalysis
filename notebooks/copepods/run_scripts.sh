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
        --n-clusters-values 2,3,4,5,10,15,20,30,40,50,100,200 \
        --tau 1

../../scripts/03_embed_and_cluster.py \
        --input data/copepods/interim/phases.parquet \
        --output-dir data/copepods/processed \
        --columns speed,curvature_angle,torsion_angle \
        --K 30 \
        --n-clusters 30 \
        --tau 1,2,5,10 \
        --groupby label

../../scripts/04_find_umap.py \
        --embedding data/copepods/processed/embedding.pkl \
        --output data/copepods/processed/umap.npy \
        --subsample 1000 \
        --cluster-centers True \
        --n-neighbors 100 \
        --min-dist 0.1                                    