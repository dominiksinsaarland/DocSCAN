path=$1
num_clusters=$2
infile=$3

mkdir $path

python src/preprocess_text_file.py --path $path --infile $infile

# compute embeddings
python src/compute_sbert_embeddings.py --infile $path/train.jsonl --outfile $path/train_embedded.pkl
#cp $path/train_embeded.pkl $path/test_embedded.pkl

# retrieve neighbors
python src/find_nearest_neighbours.py --path $path

# train DOCSCAN and predict outcome
python src/scan_classification_layer.py --path $path --num_clusters $num_clusters --num_runs 1

# stores predictions in file $path/predictions.txt
