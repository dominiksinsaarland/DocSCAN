import os
import json
from tqdm import tqdm
import csv
import argparse
import re
import random

from sklearn.model_selection import train_test_split

def preprocess_20newsgroup():
	path_train = "20newsgroup/20news-bydate-train"
	path_test = "20newsgroup/20news-bydate-test"
	path_out = "20newsgroup"

	def generate_20newsgroup_data(path, mode, path_out):
		dirs = os.listdir(path)
		outfile = open(os.path.join(path_out, mode + ".jsonl"), "w")
		for dir_ in tqdm(dirs):
			files = os.listdir(os.path.join(path, dir_))
			for filename in files:
				with open(os.path.join(path, dir_, filename), encoding="latin-1") as f:
					txt = f.read()
				txt = re.sub(r'[^\w\s]',' ',txt)
				out = {"text": txt, "label": dir_}
				json.dump(out, outfile)
				outfile.write("\n")
		outfile.close()

	generate_20newsgroup_data(path_train, "train", path_out)
	generate_20newsgroup_data(path_test, "test", path_out)

def preprocess_agnews_data():
	path_train = "CharCnn_Keras/data/ag_news_csv/train.csv"
	path_test = "CharCnn_Keras/data/ag_news_csv/test.csv"
	path_out = "ag_news"

	def generate_agnews_data(path, mode, path_out):
		outfile = open(os.path.join(path_out, mode + ".jsonl"), "w")
		with open(path) as f:
			reader = csv.reader(f, delimiter=',')
			for row in reader:
				out = {"text": row[2], "label": int(row[0])}
				json.dump(out, outfile)
				outfile.write("\n")
		outfile.close()
	generate_agnews_data(path_train, "train", path_out)
	generate_agnews_data(path_test, "test", path_out)


def preprocess_google_snippets_data():
	path_train = "google_snippets/data-web-snippets/train.txt"
	path_test = "google_snippets/data-web-snippets/test.txt"
	path_out = "google_snippets"
	def generate_google_snippets_data(path, mode, path_out):
		outfile = open(os.path.join(path_out, mode + ".jsonl"), "w")
		with open(path) as f:
			for line in f:
				line = line.strip().split()
				text = " ".join(line[:-1])
				label = line[-1]
				out = {"text": text, "label": label}
				json.dump(out, outfile)
				outfile.write("\n")
		outfile.close()

	generate_google_snippets_data(path_train, "train", path_out)
	generate_google_snippets_data(path_test, "test", path_out)


def preprocess_abstracts_group_data(): 
	filenames = ["5AbstractsGroup/Business.txt", "5AbstractsGroup/CSAI.txt", "5AbstractsGroup/Law.txt",  "5AbstractsGroup/Sociology.txt", "5AbstractsGroup/Trans.txt"]
	with open("abstracts_group/train.jsonl", "w") as train, open("abstracts_group/test.jsonl", "w") as test:
		train_examples = []
		test_examples = []
		for filename in filenames:
			label = filename[:-4]
			examples = []
			with open(filename) as f:
				for line in f:
					out = {"text": line.strip(), "label":label}
					examples.append(out)

			random.shuffle(examples)
			test_split = len(examples) // 10
			for i in examples[:test_split]:
				test_examples.append(i)
			for i in examples[test_split:]:
				train_examples.append(i)

		random.shuffle(train_examples)
		for i in test_examples:
			json.dump(i, test)
			test.write("\n")
		for i in train_examples:
			json.dump(i, train)
			test.write("\n")


	
def preprocess_dbpedia_data():
	path_train = "dbpedia/dbpedia_csv/train.csv"
	path_test = "dbpedia/dbpedia_csv/test.csv"
	path_out = "dbpedia"
	def generate_dbpedia_data(path, mode, path_out):
		outfile = open(os.path.join(path_out, mode + ".jsonl"), "w")
		X, y = [], []
		with open(path) as f:
			reader = csv.reader(f, delimiter=',')
			for row in reader:
				X.append(row[-1])
				y.append(row[0])

		for text, label in zip(X, y):
			out = {"text": text, "label": label}
			json.dump(out, outfile)
			outfile.write("\n")
		outfile.close()
	generate_dbpedia_data(path_train, "train", path_out)
	generate_dbpedia_data(path_test, "test", path_out)


preprocess_abstracts_group_data
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--experiment", default="20newsgroup", type=str, help="")
	args = parser.parse_args()

	if args.experiment == "20newsgroup":
		preprocess_20newsgroup()
	elif args.experiment == "ag_news":
		preprocess_agnews_data()
	elif args.experiment == "google_snippets":
		preprocess_google_snippets_data()
	elif args.experiment == "dbpedia":
		preprocess_dbpedia_data()
	elif args.experiment == "abstracts_group":
		preprocess_abstracts_group_data()

