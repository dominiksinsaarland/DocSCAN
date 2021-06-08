import sys
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import argparse

def load_data(infile):
	sentences = []
	labels = []
	with open(infile) as f:
		for line in f:
			line = json.loads(line)
			#print (line, line.keys())
			sentences.append(line["text"])
			labels.append(line["label"])
	return sentences, labels


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", default="", type=str, help="")
	parser.add_argument("--infile", default=None, type=str, help="", required=True)
	parser.add_argument("--outfile", default=None, type=str, help="", required=True)
	parser.add_argument("--max_seq_length", default=128, type=int, help="", required=True)

	args = parser.parse_args()
	print (args.infile)
	print (args.outfile)
	if not args.model:
		args.model = "bert-base-nli-mean-tokens"

	sentences, labels = load_data(args.infile)
	#embedder = SentenceTransformer(args.model, cache_dir="/cluster/work/lawecon/Work/dominik/transformer_models")
	embedder = SentenceTransformer(args.model)
	embedder.max_seq_length = args.max_seq_length
	corpus_embeddings = embedder.encode(sentences, batch_size=32, show_progress_bar=True)
	df = pd.DataFrame(list(zip(sentences, labels, corpus_embeddings)), columns=["sentence", "label", "embeddings"])
	df.to_pickle(args.outfile)

