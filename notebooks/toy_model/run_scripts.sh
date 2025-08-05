for filename in only_persistent stereotypes2 stereotypes4
do
./../../scripts/01_make_phases.py \
        --input-path ../../data/toy_model/raw/ \
        --input-name $filename \
        --output-path ../../data/toy_model/interim/ \
        --smooth True \
        --window 5 \
        --polyorder 3  \
        --max-length 997
done

for filename in only_persistent stereotypes2 stereotypes4
do
  ./../../scripts/02_compute_entropy_production.py \
      --input-path ../../data/toy_model/interim/ \
      --input-name phases_$filename \
      --output-path ../../data/toy_model/interim/ \
      --columns-trans x,y,z \
      --K-values 3,5,10,20,30,40,50 \
      --n-clusters-values 2,3,4,5,10,25,50,100,150,250,500,1000,2000 \
      --tau 1
done

for filename in only_persistent stereotypes2 stereotypes4
do
./../../scripts/03_embed_and_cluster.py \
        --input-path ../../data/toy_model/interim/ \
        --input-name $filename \
        --output-path ../../data/toy_model/processed \
        --columns-trans x,y,z \
        --K 30 \
        --n-clusters 1000 \
        --tau-values 1,2,5,10,15,20,30,40,50,75,100,125,150,175,200 \
        --groupby label
done

for filename in only_persistent stereotypes2 stereotypes4
do
./../../scripts/04_find_umap.py \
        --input-path ../../data/toy_model/processed/ \
        --input-name $filename \
        --output-path ../../data/toy_model/processed/ \
        --subsample 1000 \
        --cluster-centers True \
        --n-neighbors 100 \
        --min-dist 0.1
done
