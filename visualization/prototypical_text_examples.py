import os
import pandas as pd
import numpy as np
import argparse
import json
from tqdm import tqdm

def load_clusters(fn):
	with open(fn) as f:
		clusters = [i.strip() for i in f]
	return clusters

def load_probabilities(fn):
	with open(fn) as f:
		probabilities = [list(map(float, i.strip().split())) for i in f]
	return probabilities

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--path", default="20newsgroup", type=str, help="", required=True)
	parser.add_argument("--outfile", default="prototypical_examples_by_clusters", type=str, help="")
	args = parser.parse_args()


	if os.path.exists(os.path.join(args.path, "test_embedded.pkl")):
		filename = os.path.join(args.path, "test_embedded.pkl")
	else:
		filename = os.path.join(args.path, "train_embedded.pkl")

	if os.path.exists(os.path.join(args.path, "label2id.json")):
		with open(os.path.join(args.path, "label2id.json")) as f:		
			label2id = json.load(f)
	else:
		label2id = {}


	df = pd.read_pickle(filename)
	clusters = load_clusters(os.path.join(args.path, "predictions.txt"))
	probabilities = load_probabilities(os.path.join(args.path, "probabilities.txt"))
	df["clusters"] = clusters
	df["probabilities"] = probabilities

	with open(os.path.join(args.path, args.outfile), "w") as outfile:
		for topic in tqdm(np.unique(clusters)):
			df_topic = df[df["clusters"] == topic]
			if label2id:
				probabilites = [i[label2id[topic]] for i in df["probabilities"]]
			else:
				probabilites = [i[label2id[int(topic)]] for i in df["probabilities"]]
			indices = np.argsort(probabilites)[-10:]
			for i in indices:
				#print (probabilities[i], "--", df.iloc[i]["sentence"])
				outfile.write(topic + "\t" + df.iloc[i]["sentence"] + "\n")
			#input("")


			"""
			most_probable_topics = np.argsort(df_topic["probabilities"])[-5:]
			for index in most_probable_topics:
				print (topic, "--", 
				sentence = df_topic.iloc[index]["sentence"]
				outfile.write(topic + "\t" + sentence + "\n")
			"""

