path=$1
infile=$2
min_clusters=$3
max_clusters=$4
step_size=$5

mkdir $path

python src/preprocess_text_file.py --path $path --infile $infile

# compute embeddings
python src/compute_sbert_embeddings.py --infile $path/train.jsonl --outfile $path/train_embedded.pkl
#cp $path/train_embeded.pkl $path/test_embedded.pkl

# retrieve neighbors
python src/find_nearest_neighbours.py --path $path

# train DOCSCAN and automatically search for best number of clusters

python src/scan_distortion_scores.py --path $path --min_clusters $min_clusters --max_clusters $max_clusters --stepsize $step_size


# train DOCSCAN and predict outcome
python src/scan_classification_layer.py --path $path --num_clusters $num_clusters --num_runs 1

# generates an elbow plot and saves the plot in $path/optimal_number_of_clusters.png

# if we find a "knee" in the plot, we re-train the model automatically using the optimal amount of clusters; We then assign clusters to each sentence in $path/docscan_clusters.csv


# the remainder is visualization; for now, we plot the prototypical examples of each cluster

# find prototypal examples of each cluster
python visualization/prototypical_text_examples.py --path $path

# and we create tf-idf based wordclouds for each cluster

python visualization/word_clouds.py --path $path
