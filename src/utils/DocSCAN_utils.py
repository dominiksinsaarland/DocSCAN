import argparse
import os
import json
import pandas as pd
import torch
import random
import gc

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
from sklearn.metrics.pairwise import pairwise_distances

import matplotlib.pyplot as plt

#from visualizations import generate_word_clouds

class DocScanDataset(Dataset):
	def __init__(self, neighbor_df, embeddings, test_embeddings="", mode="train"):
		self.neighbor_df = neighbor_df
		self.embeddings = embeddings
		self.mode = mode
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		if mode == "train":
			self.examples = self.load_data()
		elif mode == "predict":
			self.examples = test_embeddings

	def load_data(self):
		examples = []
		for i,j in zip(self.neighbor_df["anchor"], self.neighbor_df["neighbor"]):
			examples.append((i,j))
		random.shuffle(examples)
		return examples

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, item):
		if self.mode == "train":
			anchor, neighbor = self.examples[item]
			sample = {"anchor": anchor, "neighbor": neighbor}
		elif self.mode == "predict":
			anchor = self.examples[item]
			sample = {"anchor": anchor}
		return sample
	def collate_fn(self, batch):
		anchors = torch.tensor([i["anchor"] for i in batch])
		out = self.embeddings[anchors].to(self.device)
		neighbors = torch.tensor([i["anchor"] for i in batch])
		out_2 = self.embeddings[neighbors].to(self.device)
		return {"anchor": out, "neighbor": out_2}

	def collate_fn_predict(self, batch):
		out = torch.vstack([i["anchor"] for i in batch]).to(self.device)
		return {"anchor": out}

class DocScanModel(torch.nn.Module):
	def __init__(self, num_labels, dropout, hidden_dim=768):
		super(DocScanModel, self).__init__()
		self.num_labels = num_labels
		self.classifier = torch.nn.Linear(hidden_dim, num_labels)
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		#self.device = "cpu"
		self.dropout = dropout

	def forward(self, feature):
		if self.dropout is not None:
			dropout = torch.nn.Dropout(p=self.dropout)
			feature = dropout(feature)
		output = self.classifier(feature)
		return output
