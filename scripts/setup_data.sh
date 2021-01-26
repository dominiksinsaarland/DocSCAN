mkdir 20newsgroup
mkdir ag_news
mkdir dbpedia
mkdir google_snippets
mkdir abstracts_group

tar -xf 20news-bydate.tar.gz -C 20newsgroup/
tar -xf data-web-snippets.tar.gz -C google_snippets/
tar -xf dbpedia_csv.tar.gz -C dbpedia/

# preprocess different datasets
python src/preprocess.py --experiment 20newsgroup
python src/preprocess.py --experiment ag_news
python src/preprocess.py --experiment dbpedia
python src/preprocess.py --experiment google_snippets
python src/preprocess.py --experiment abstracts_group

# compute embeddings

python src/compute_sbert_embeddings.py --infile 20newsgroup/train.jsonl --outfile 20newsgroup/train_embedded.pkl
python src/compute_sbert_embeddings.py --infile 20newsgroup/test.jsonl --outfile 20newsgroup/test_embedded.pkl

python src/compute_sbert_embeddings.py --infile ag_news/train.jsonl --outfile ag_news/train_embedded.pkl
python src/compute_sbert_embeddings.py --infile ag_news/test.jsonl --outfile ag_news/test_embedded.pkl

python src/compute_sbert_embeddings.py --infile dbpedia/train.jsonl --outfile dbpedia/train_embedded.pkl
python src/compute_sbert_embeddings.py --infile dbpedia/test.jsonl --outfile dbpedia/test_embedded.pkl

python src/compute_sbert_embeddings.py --infile google_snippets/train.jsonl --outfile google_snippets/train_embedded.pkl
python src/compute_sbert_embeddings.py --infile google_snippets/test.jsonl --outfile google_snippets/test_embedded.pkl

python src/compute_sbert_embeddings.py --infile abstracts_group/train.jsonl --outfile abstracts_group/train_embedded.pkl
python src/compute_sbert_embeddings.py --infile abstracts_group/test.jsonl --outfile abstracts_group/test_embedded.pkl



# models trained on detecting paraphrases perform worse

#python src/compute_sbert_embeddings.py --infile google_snippets/train.jsonl --outfile google_snippets/train_embedded_paraphrases.pkl --model paraphrase-distilroberta-base-v1
#python src/compute_sbert_embeddings.py --infile google_snippets/test.jsonl --outfile google_snippets/test_embedded_paraphrases.pkl --model paraphrase-distilroberta-base-v1
#python src/compute_sbert_embeddings.py --infile 20newsgroup/train.jsonl --outfile 20newsgroup/train_embedded_paraphrases.pkl --model paraphrase-distilroberta-base-v1
#python src/compute_sbert_embeddings.py --infile 20newsgroup/test.jsonl --outfile 20newsgroup/test_embedded_paraphrases.pkl --model paraphrase-distilroberta-base-v1
