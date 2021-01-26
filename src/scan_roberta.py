
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
	def __init__(self, filename, tokenizer, mode="train", translation_fn=None):
		self.filename = filename
		self.mode = mode
		self.tokenizer = tokenizer
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.translation_fn= translation_fn
		if mode == "train":
			self.examples = self.load_data()
		elif mode == "val":
			self.examples = self.load_val_data()
	def load_data(self):
		examples = []
		labels = []
		with open(self.filename) as f:
			for i, line in enumerate(f):
				line = json.loads(line)
				anchor, neighbour = line["anchor"], line["neighbour"]
				examples.append((anchor, neighbour))
		random.shuffle(examples)
		#examples = examples[:250000]
		return examples

	def load_val_data(self):
		df = pd.read_pickle(self.filename)
		labels=np.unique(df["label"])
		labels_map = {label:i for i, label in enumerate(labels)}
		examples = []
		for sent, label in zip(df["sentence"], df["label"]):
			examples.append((sent, labels_map[label]))
		return examples

	def __len__(self):
		return len(self.examples)
	def __getitem__(self, item):
		#print (item)
		if self.mode == "train":
			#anchor, neighbour, label = self.examples[item]
			anchor, neighbour = self.examples[item]
			sample = {"anchor": anchor, "neighbour": neighbour}
		elif self.mode == "val":
			anchor, label = self.examples[item]
			sample = {"anchor": anchor, "label": label}
		return sample
	def collate_fn(self, batch):
		out = self.tokenizer([i["anchor"] for i in batch], return_tensors="pt", padding=True, truncation=True).to(self.device)
		out_2 = self.tokenizer([i["neighbour"] for i in batch], return_tensors="pt", padding=True, truncation=True).to(self.device)
		return {"anchor": out, "neighbour": out_2}

	def collate_fn_val(self, batch):
		out = self.tokenizer([i["anchor"] for i in batch], return_tensors="pt", padding=True, truncation=True).to(self.device)
		labels = torch.tensor([i["label"] for i in batch]).to(self.device)
		return {"anchor": out, "label": labels}

class SCAN_model(torch.nn.Module):
	def __init__(self, model, num_labels, dropout):
		super(SCAN_model, self).__init__()
		self.model = model
		self.num_labels = num_labels
		try:
			self.classifier = torch.nn.Linear(self.model.config.dim, num_labels)
		except:
			self.classifier = torch.nn.Linear(self.model.config.hidden_size, num_labels)

		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.dropout = dropout
	def forward(self, feature):
		output = self.model(**feature)
		hidden_state = output[0]
		pooled_output = hidden_state[:, 0]
		output = self.classifier(pooled_output)
		return output

	def get_predictions(self, dataset):
		predictions, probs, targets = [], [], []
		epoch_iterator = tqdm(dataset)
		for i, batch in enumerate(epoch_iterator):
			with torch.no_grad():
				self.model.eval()
				output_i = self.forward(batch["anchor"])
				probs.extend(torch.nn.functional.softmax(output_i, dim=1).cpu().numpy())
				predictions.extend(torch.argmax(output_i, dim=1).cpu().numpy())
				try:
					targets.extend(batch["label"].cpu().numpy())
				except:
					pass
		out = {'predictions': predictions, 'probabilities': probs, 'targets': targets}
		return out

	def early_stopping(self, dataset):
		predictions, probs, targets = [], [], []
		epoch_iterator = tqdm(dataset)
		for i, batch in enumerate(epoch_iterator):
			with torch.no_grad():
				self.model.eval()
				output_i = self.forward(batch["anchor"])
				probs.extend(torch.nn.functional.softmax(output_i, dim=1).cpu().numpy())
				predictions.extend(torch.argmax(output_i, dim=1).cpu().numpy())
		print (np.unique(predictions, return_counts=True))
		if len(np.unique(predictions)) < num_classes:
			return True
		else:
			return False

	def compute_embeddings(self, dataset):
		embeddings = []
		epoch_iterator = tqdm(dataset)
		for i, batch in enumerate(epoch_iterator):
			with torch.no_grad():
				output = self.model(**batch["anchor"])  # (bs, dim)
				hidden_state = output[0]
				pooled_output = hidden_state[:, 0]
				embeddings.extend(pooled_output.cpu().numpy())
		return embeddings

	def save(self, output_path, tokenizer):
		self.model.save_pretrained(output_path)
		tokenizer.save_pretrained(output_path)
		torch.save(self.classifier.state_dict(), os.path.join(output_path, "classification layer"))



def load_model_and_tokenizer(model_name_or_path, num_classes, dropout, from_finetuned=False):
	if not from_finetuned:
		# DistilBertTokenizerFast, DistilBertModel
		if model_name_or_path == "distilbert-base-uncased":
			tokenizer = DistilBertTokenizerFast.from_pretrained(model_name_or_path) 
			model = DistilBertModel.from_pretrained(model_name_or_path)

		else:
			tokenizer = RobertaTokenizerFast.from_pretrained(model_name_or_path) 
			model = RobertaModel.from_pretrained(model_name_or_path)

		model = SCAN_model(model, num_classes, dropout)
		return model, tokenizer
	else:
		tokenizer = RobertaTokenizerFast.from_pretrained(model_name_or_path) 
		model = RobertaModel.from_pretrained(model_name_or_path)
		model = SCAN_model(model, num_classes, dropout)
		model.classifier.load_state_dict(torch.load(os.path.join(model_name_or_path, "classification layer"),  map_location=torch.device(device)))
		return model, tokenizer
		


def evaluate(model, val_dataloader):
	model.zero_grad()
	out = model.get_predictions(val_dataloader)
	#out = model.get_predictions(val_dataloader)
	hungarian_match_metrics = hungarian_evaluate(np.array(out["targets"]), np.array(out["predictions"]))
	print (fn_val, hungarian_match_metrics)
	return hungarian_match_metrics
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--experiment", type=str, default="20_newsgroup", help="")
	parser.add_argument("--max_grad_norm", default=1.0, type=float, help="")
	parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
		        help="Number of updates steps to accumulate before performing a backward/update pass.")
	parser.add_argument("--batch_size", default=4, type=int,
		        help="Batch size per GPU/CPU for training.")
	parser.add_argument("--num_epochs", default=1, type=int,
		        help="Total number of training epochs to perform.")
	parser.add_argument("--learning_rate", default=2e-5, type=float,
		        help="The initial learning rate for Adam.")
	parser.add_argument("--adam_epsilon", default=1e-8, type=float,
		        help="Epsilon for Adam optimizer.")
	parser.add_argument("--weight_decay", default=0.0, type=float,
		        help="Weight decay if we apply some.")
	parser.add_argument("--only_eval", action="store_true",
		        help="Whether to run evaluation.")
	parser.add_argument("--get_embeddings", action="store_true",
		        help="Whether to run evaluation.")
	parser.add_argument("--i", type=str, default=None, help="")
	parser.add_argument("--use_translations", action="store_true",
		        help="Whether to run evaluation.")
	parser.add_argument("--iteration", type=str, default=None, help="")
	parser.add_argument("--entropy_weight", default=2, type=float,
		        help="Batch size per GPU/CPU for training.")
	parser.add_argument("--model_name", default="roberta-base", type=str,
		        help="Batch size per GPU/CPU for training.")
	parser.add_argument("--dropout", default=None, type=float,
		        help="Batch size per GPU/CPU for training.")



	args = parser.parse_args()

	args = parser.parse_args()
	if args.experiment == "20newsgroup":
		num_classes = 20
		path = "20newsgroup"
	elif args.experiment == "ag_news":
		path = "ag_news"
		num_classes = 4
	elif args.experiment == "google_snippets":
		path = "google_snippets"
		num_classes = 8
	elif args.experiment == "abstracts_group":
		path = "abstracts_group"
		num_classes = 5
	elif args.experiment == "dbpedia":
		path = "dbpedia"
		num_classes = 14
	elif args.experiment == "imdb":
		path = "imdb"
		num_classes = 2
	elif args.experiment == "ner":
		path = "CoNLL-2003"
		num_classes = 4
	elif args.experiment == "pos_tagging":
		path = "brown_POS"
		num_classes = 12
	else:
		print ("experiment unknown")
		sys.exit(0)


	fn_val = os.path.join(path, "test_embedded.pkl")
	fn_train = os.path.join(path, "train_neighbours.jsonl")

	device = "cuda" if torch.cuda.is_available() else "cpu"

	# CUDNN
	torch.backends.cudnn.benchmark = True


	#args.model_name = 'distilbert-base-uncased'

	model, tokenizer = load_model_and_tokenizer(args.model_name, num_classes, args.dropout)
	model.to(device)

	train_dataset = ScanDataset(fn_train, tokenizer, mode="train")
	val_dataset = ScanDataset(fn_val, tokenizer, mode="val")

	train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn = train_dataset.collate_fn, batch_size=args.batch_size)
	val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, collate_fn = val_dataset.collate_fn_val, batch_size=args.batch_size)

	warmup_steps = 0
	t_total = int(len(train_dataloader) * args.num_epochs / args.gradient_accumulation_steps)

	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
	{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
	{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

	optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
	# Loss function
	criterion = SCANLoss(entropy_weight=args.entropy_weight)

	criterion.to(device)
	model.zero_grad()

	# Main loop
	print('Starting main loop')
	if args.i:
		outfile = open(os.path.join(path, "scan_results_" + args.i + ".txt"), "w")
	else:
		outfile = open(os.path.join(path, "eval_results.txt"), "w")
	for epoch in range(args.num_epochs):
		model.train()
		bar_desc = "Epoch %d of %d | Iteration" % (epoch + 1, args.num_epochs)
		epoch_iterator = tqdm(train_dataloader, desc=bar_desc)
		print('Train ...')
		for step, batch in enumerate(epoch_iterator):
			anchor, neighbour = batch["anchor"], batch["neighbour"]
			anchors_output, neighbors_output = model(anchor), model(neighbour)
			total_loss, consistency_loss, entropy_loss = criterion(anchors_output, neighbors_output)
			if args.gradient_accumulation_steps > 1:
				total_loss = total_loss / args.gradient_accumulation_steps

			total_loss.backward()
			if (step + 1) % args.gradient_accumulation_steps == 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
				optimizer.step()
				scheduler.step()  # Update learning rate schedule
				model.zero_grad()

		evaluate(model, val_dataloader)

	# save if not exit through early stopping
	if not flag_early_stopping:
		model.save(output_path, tokenizer)
	# else, reload from last checkpoint
	else:
		model, tokenizer = load_model_and_tokenizer(output_path, num_classes, args.dropout, from_finetuned=True)
		model.to(device)
	# evaluate dev set
	hungarian_match_metrics = evaluate(model, val_dataloader)

	json.dump(args.__dict__, outfile)
	for i,j in hungarian_match_metrics.items():
		outfile.write("epoch: " + str(epoch + 1) + "\n" + i + " " + str(j) + "\n" + "\n" + "num_steps " + str(step) + " ")
	outfile.close()

	"""

	# evaluate train set
	train_dataset = ScanDataset(fn_train_pickle, tokenizer, mode="val")
	train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=False, collate_fn = train_dataset.collate_fn_val, batch_size=args.batch_size)
	out = model.get_predictions(train_dataloader)


	if args.iteration:
		outfile_name = os.path.join(path, "predictions_trainset_iteration_" + args.iteration + ".txt")
	else:
		outfile_name = os.path.join(path, "predictions_trainset.txt")

	with open(outfile_name, "w") as outfile:
		for i in out["predictions"]:
			outfile.write(str(i) + "\n")

	print (np.unique(out["predictions"], return_counts=True), np.unique(out["targets"], return_counts=True))
	hungarian_match_metrics = hungarian_evaluate(np.array(out["targets"]), np.array(out["predictions"]))
	print (fn_val, hungarian_match_metrics)

	if args.iteration:
		outfile_name = os.path.join(path, "eval_results_trainset_iteration_" + args.iteration + ".txt")
	else:
		outfile_name = os.path.join(path, "eval_results_trainset.txt")
	with open(outfile_name, "w") as outfile:
		for i,j in hungarian_match_metrics.items():
			outfile.write(i + " " + str(j) + "\n" + "\n")

	# get embeddings for next round of SCAN
	out_df = pd.read_pickle(fn_train_pickle)
	out = model.compute_embeddings(train_dataloader)
	out_df["embeddings"] = out
	if args.iteration:
		out_df.to_pickle(os.path.join(path, "train_embedded_iteration_" + str(int(args.iteration) + 1) + ".pkl"))
	else:
		out_df.to_pickle(os.path.join(path, "train_embedded_iteration_1.pkl"))

	"""


# bsub -n 1 -R "rusage[mem=25600,ngpus_excl_p=1]" python src/scan.py 
# bsub -n 1 -W 1440 -R "rusage[mem=25600,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python src/scan.py 

