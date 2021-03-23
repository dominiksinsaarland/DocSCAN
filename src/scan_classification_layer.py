import argparse
import os
import json
import pandas as pd
import torch
import random
import gc

from losses import SCANLoss
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import DistilBertTokenizerFast, DistilBertModel
from transformers import RobertaTokenizerFast, RobertaModel
from scipy.optimize import linear_sum_assignment

from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from utils import *


class ScanDataset(Dataset):
	def __init__(self, filename, mode="train", translation_fn=None):
		self.filename = filename
		self.mode = mode
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.translation_fn= translation_fn
		if mode == "train":
			self.examples = self.load_data()
		elif mode == "val":
			self.examples = self.load_val_data()
	def load_data(self):
		examples = []

		df = pd.read_pickle(self.filename)
		anchors = df["anchor"].tolist()
		neighbours = df["neighbour"].tolist()
		labels = df["label"].tolist()
		counts = defaultdict(int)
		for i,j in zip(anchors, neighbours):
			examples.append((i,j))

		random.shuffle(examples)
		return examples


	def load_val_data(self):
		df = pd.read_pickle(self.filename)
		labels=np.unique(df["label"])
		labels_map = {label:i for i, label in enumerate(labels)}
		examples = []
		anchors = df["embeddings"].tolist()
		label = df["label"].tolist()
		for i,j in zip(anchors, label):
			examples.append((i,labels_map[j]))
		return examples

	def __len__(self):
		return len(self.examples)
	def __getitem__(self, item):

		if self.mode == "train":
			anchor, neighbour = self.examples[item]
			sample = {"anchor": anchor, "neighbour": neighbour}
		elif self.mode == "val":
			anchor, label = self.examples[item]
			sample = {"anchor": anchor, "label": label}
		return sample
	def collate_fn(self, batch):
		out = torch.tensor([i["anchor"] for i in batch])
		out_2 = torch.tensor([i["neighbour"] for i in batch])
		return {"anchor": out, "neighbour": out_2}

	def collate_fn_val(self, batch):
		out = torch.tensor([i["anchor"] for i in batch])
		labels = torch.tensor([i["label"] for i in batch]).to(self.device)
		return {"anchor": out, "label": labels}

class SCAN_model(torch.nn.Module):
	def __init__(self, num_labels, dropout):
		super(SCAN_model, self).__init__()
		self.num_labels = num_labels
		self.classifier = torch.nn.Linear(768, num_labels)
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.dropout = dropout
	def forward(self, feature):
		if self.dropout is not None:
			dropout = torch.nn.Dropout(p=self.dropout)
			feature = dropout(feature)
		output = self.classifier(feature)
		return output

	def get_predictions(self, dataset):
		predictions, probs, targets = [], [], []
		epoch_iterator = tqdm(dataset)
		for i, batch in enumerate(epoch_iterator):
			self.classifier.eval()
			output_i = self.forward(batch["anchor"])
			probs.extend(torch.nn.functional.softmax(output_i, dim=1).cpu().numpy())
			predictions.extend(torch.argmax(output_i, dim=1).cpu().numpy())
			try:
				targets.extend(batch["label"].cpu().numpy())
			except:
				pass
		out = {'predictions': predictions, 'probabilities': probs, 'targets': targets}
		return out
		
def evaluate(model, val_dataloader):
	with torch.no_grad():
		out = model.get_predictions(val_dataloader)
	hungarian_match_metrics = hungarian_evaluate(np.array(out["targets"]), np.array(out["predictions"]))
	cm = hungarian_match_metrics["confusion matrix"]
	clf_report = hungarian_match_metrics["classification_report"]
	print (fn_val, hungarian_match_metrics)
	del hungarian_match_metrics["classification_report"]

	del hungarian_match_metrics["confusion matrix"]
	print (cm, "\n", clf_report)
	print (fn_val, hungarian_match_metrics)
	print ("ACCURACY", np.round(hungarian_match_metrics["ACC"], 3))
	return hungarian_match_metrics
	

def predict(model, val_dataloader, path):
	with torch.no_grad():
		out = model.get_predictions(val_dataloader)
	#
	hungarian_match_metrics = hungarian_evaluate(np.array(out["targets"]), np.array(out["predictions"]))
	fn_val = os.path.join(path, "test_embedded.pkl")
	df = pd.read_pickle(fn_val)
	labels=np.unique(df["label"])
	label2id = {label:i for i, label in enumerate(labels)}
	id2label = {i:j for j,i in label2id.items()}

	with open(os.path.join(path, "predictions.txt"), "w") as outfile:
		for i in hungarian_match_metrics["reordered_preds"]:
			outfile.write(str(id2label[i]) + "\n")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--path", type=str, default="20newsgroup", help="")
	parser.add_argument("--max_grad_norm", default=1.0, type=float, help="")
	parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
		        help="Number of updates steps to accumulate before performing a backward/update pass.")
	parser.add_argument("--batch_size", default=64, type=int,
		        help="Batch size per GPU/CPU for training.")
	parser.add_argument("--num_epochs", default=3, type=int,
		        help="Total number of training epochs to perform.")
	parser.add_argument("--learning_rate", default=0.001, type=float,
		        help="The initial learning rate for Adam.")
	parser.add_argument("--adam_epsilon", default=1e-8, type=float,
		        help="Epsilon for Adam optimizer.")
	parser.add_argument("--entropy_weight", default=2, type=float,
		        help="Batch size per GPU/CPU for training.")
	parser.add_argument("--entropy_term", type=str, default="entropy", help="")

	parser.add_argument("--dropout", default=0.1, type=float,
		        help="Batch size per GPU/CPU for training.")
	parser.add_argument("--weight_decay", default=0.0, type=float,
		        help="Weight decay if we apply some.")
	parser.add_argument("--num_clusters", default=1, type=int,
		        help="number of clusters, if not defined, we set it to the number of classes in the training set")


	args = parser.parse_args()

	fn_val = os.path.join(args.path, "test_embedded.pkl")
	fn_train = os.path.join(args.path, "train_neighbours_embeddings.pkl")

	if args.num_clusters == 1:
		df = pd.read_pickle(fn_train)
		num_classes = len(np.unique(df["label"]))
	else:
		num_classes = args.num_clusters

	device = "cuda" if torch.cuda.is_available() else "cpu"

	# CUDNN
	torch.backends.cudnn.benchmark = True

	train_dataset = ScanDataset(fn_train, mode="train")
	val_dataset = ScanDataset(fn_val, mode="val")

	train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn = train_dataset.collate_fn, batch_size=args.batch_size)
	val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, collate_fn = val_dataset.collate_fn_val, batch_size=args.batch_size)

	results = []

	for _ in range(5):
		model = SCAN_model(num_classes, args.dropout)
		model.to(device)

		optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-8)
		# Loss function
		criterion = SCANLoss(entropy_weight=args.entropy_weight, entropy=args.entropy_term, experiment=args.path)
		criterion.to(device)
		model.zero_grad()
		model.train()

		train_iterator = range(int(args.num_epochs))
		for epoch in train_iterator:
			bar_desc = "Epoch %d of %d | Iteration" % (epoch + 1, len(train_iterator))
			epoch_iterator = tqdm(train_dataloader, desc=bar_desc)
			for step, batch in enumerate(epoch_iterator):
				optimizer.zero_grad()
				model.zero_grad()

				anchor, neighbour = batch["anchor"], batch["neighbour"]
				anchors_output, neighbors_output = model(anchor), model(neighbour)
				total_loss, consistency_loss, entropy_loss = criterion(anchors_output, neighbors_output)
				total_loss.backward()
				optimizer.step()

		acc = evaluate(model, val_dataloader)["ACC"]
		results.append(acc)

	with open(os.path.join(args.path, "scan_results_classification_layer.txt"), "w") as outfile:
		config_string = "lr: " + str(args.learning_rate) + " num_epochs: " +str(args.num_epochs) + " batch_size: " + str(args.batch_size) + " dropout: " + str(args.dropout)
		print (np.round(np.mean(results), 3), np.std(results))
		#outfile.write(config_string + "\n")
		#outfile.write(str(results) + "\n")
		outfile.write("accuracy: " + str(np.round(np.mean(results), 3)) + "\nstandard error: " + str(np.round(np.std(results), 3)))
		predict(model, val_dataloader, args.path)

