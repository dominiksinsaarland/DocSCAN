from memory import MemoryBank
import pandas as pd
import torch
import numpy as np
import json
import os
import argparse


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--path", type=str, default="20newsgroup", help="")
	parser.add_argument("--topk", type=int, default=5, help="")
	parser.add_argument("--train_file", type=str, default="")
	parser.add_argument("--outfile", type=str, default="")
	parser.add_argument("--show_accuracies", action="store_true", help="")
	args = parser.parse_args()


	if args.train_file and args.outfile:
		fn_train = args.train_file
		path_out = args.outfile
	else:
		fn_train = os.path.join(args.path, "train_embedded.pkl")
		path_out = os.path.join(args.path, "train_neighbours.jsonl")

	df_train = pd.read_pickle(fn_train)
	X_train = np.array(df_train["embeddings"].tolist())

	print (np.shape(X_train))
	feature_dim = np.shape(X_train[-1])



	X_train = torch.tensor(X_train)
	labels=np.unique(df_train["label"])
	labels_map = {label:i for i, label in enumerate(labels)}
	targets = torch.tensor([labels_map[i] for i in df_train["label"]])

	num_classes = len(labels)
	memory_bank_base = MemoryBank(X_train, targets, len(X_train), 
		                        feature_dim,
		                        num_classes)


	if args.show_accuracies:
		acc_train = []
		results_file = os.path.join(args.path, "results_nearest_neighbours.txt")
		with open(results_file, "w") as f:
			for topk in [1, 5, 50, 100]:
				indices, acc = memory_bank_base.mine_nearest_neighbors(topk, show_eval=True)
				res = 'Accuracy of top-%d nearest neighbors on train set is %.2f' %(topk, 100*acc)
				f.write(res + "\n")
				acc_train.append(acc)
		print ("training set accuracies", [np.round(i * 100, 1) for i in acc_train])


	indices, acc = memory_bank_base.mine_nearest_neighbors(args.topk, show_eval=False)
	#print('Accuracy of top-%d nearest neighbors on train set is %.2f' %(args.topk, 100*acc))


	sentences = df_train["sentence"].tolist()
	with open(path_out, "w") as outfile:
		for index, label in zip(indices, df_train["label"]): 
			anchor = index[0]
			neighbours = index[1:]
			for neighbour in neighbours:
				out = {"anchor": sentences[anchor], "neighbour": sentences[neighbour], "label":labels_map[label]}
				json.dump(out, outfile)
				outfile.write("\n")
	
	examples = []
	for index, label in zip(indices, df_train["label"]): 
		anchor = index[0]
		neighbours = index[1:]
		for neighbour in neighbours:
			examples.append([X_train[anchor].tolist(), X_train[neighbour].tolist(), labels_map[label]])

	df = pd.DataFrame(examples, columns=["anchor", "neighbour", "label"])
	df.to_pickle(os.path.join(args.path, "train_neighbours_embeddings.pkl"))

