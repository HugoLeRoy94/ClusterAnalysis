for filename in fed_track starve_track
do
./../../scripts/01_make_phases.py \
        --input-path ../../data/choano/raw/ \
        --input-name $filename \
        --output-path ../../data/choano/interim/ \
        --extension parquet \
        --groupby TRACK_ID \
        --sortby FRAME \
        --cols POSITION_X,POSITION_Y,POSITION_Z \
        --smooth True \
        --window 5 \
        --polyorder 3  \
        --max-length 100 
done

for filename in fed_track starve_track
do
  ./../../scripts/02_compute_entropy_production.py \
      --input-path ../../data/choano/interim/ \
      --extension parquet \
      --input-name phases_$filename \
      --output-path ../../data/choano/interim/ \
      --columns-trans x,y,z \
      --K-values 3,5,10,20,30,40,50 \
      --n-clusters-values 2,3,4,5,10,25,50,100,150,250,500,1000,2000 \
      --tau 1
done


for filename in fed_track starve_track
do
./../../scripts/03_embed_and_cluster.py \
        --input-path ../../data/choano/interim/ \
        --input-name $filename \
        --output-path ../../data/choano/processed \
        --columns-trans x,y,z \
        --K 30 \
        --n-clusters 100 \
        --tau-values 1,2,5,10,15,20,30,40,50,75,100,125,150,175,200 \
        --groupby label
done

for filename in fed_track starve_track
do
./../../scripts/04_find_umap.py \
        --input-path ../../data/choano/processed/ \
        --input-name $filename \
        --output-path ../../data/choano/processed/ \
        --cluster-centers True \
        --n-neighbors 10 \
        --min-dist 0.01
done