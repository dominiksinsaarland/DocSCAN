# tf-idf
python src/kmeans.py --path 20newsgroup --features tfidf
python src/kmeans.py --path ag_news --features tfidf
python src/kmeans.py --path dbpedia --features tfidf
python src/kmeans.py --path google_snippets --features tfidf
python src/kmeans.py --path abstracts_group --features tfidf

# SBERT embeddings
python src/kmeans.py --path 20newsgroup
python src/kmeans.py --path ag_news
python src/kmeans.py --path dbpedia
python src/kmeans.py --path google_snippets
python src/kmeans.py --path abstracts_group

# inspect results
cat google_snippets/mean_kmeans_sbert.txt
cat google_snippets/mean_kmeans_tfidf.txt



