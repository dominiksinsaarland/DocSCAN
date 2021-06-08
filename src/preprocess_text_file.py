import os
import json
from tqdm import tqdm
import csv
import argparse
import re
import random

def preprocess(path, infile):
	covered = set()
	with open(os.path.join(path, "train.jsonl"), "w") as train, open(os.path.join(path, "test.jsonl"), "w") as test, open(infile) as f:
		for line in f:
			line = line.strip()
			if line in covered:
				continue
			covered.add(line)
			out = {"text": line, "label": "dummy label"}
			json.dump(out, train)
			json.dump(out, test)
			train.write("\n")
			test.write("\n")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--path", default="20newsgroup", type=str, help="", required=True)
	parser.add_argument("--infile", type=str, required=True)
	args = parser.parse_args()
	preprocess(args.path, args.infile)


