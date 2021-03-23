path=$1
num_clusters=$2

# compute embeddings
python src/compute_sbert_embeddings.py --infile $path/train.jsonl --outfile $path/train_embedded.pkl
python src/compute_sbert_embeddings.py --infile $path/test.jsonl --outfile $path/test_embedded.pkl

# baselines
# kmeans tf-idf baseline
python src/kmeans.py --path $path --features tfidf
# kmeans sbert baseline
python src/kmeans.py --path $path --features sbert

# retrieve neighbors
 python src/find_nearest_neighbours.py --path $path

# train DOCSCAN
python src/scan_classification_layer.py --path $path --num_clusters $num_clusters

# print results

printf "\nkmeans with tf-idf features: mean accuracy over 5 runs\n"
cat $path/mean_kmeans_tfidf.txt

printf "\nkmeans with sbert embeddings: mean accuracy over 5 runs\n"
cat $path/mean_kmeans_sbert.txt

printf "\ndocscan: mean accuracy over 5 runs\n"
cat $path/scan_results_classification_layer.txt
printf "\n"
