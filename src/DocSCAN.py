import sys, os, json, argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
from memory import MemoryBank
import torch
from DocSCAN_utils import DocScanDataset, DocScanModel
from losses import SCANLoss
from kneelocator import KneeLocator
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
from word_clouds import generate_word_clouds
from spacy.lang.en import English

class DocSCANPipeline():
	def __init__(self, args):
		self.args = args
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		os.makedirs(self.args.outpath, exist_ok=True)

	def load_data(self):
		if self.args.data_format == "from_txt":
			with open(self.args.infile) as f:
				sentences = [line.strip() for line in f]
			df = pd.DataFrame(sentences, columns=["sentence"])
		elif args.data_format == "from_csv":
			df = pd.read_csv(self.args.infile)
		return df
			
	def embedd_sentences(self, sentences):
		embedder = SentenceTransformer(self.args.sbert_model)
		embedder.max_seq_length = self.args.max_seq_length
		corpus_embeddings = embedder.encode(sentences, batch_size=32, show_progress_bar=True)
		return corpus_embeddings

	def create_neighbor_dataset(self):
		indices = self.memory_bank.mine_nearest_neighbors(self.args.topk, show_eval=False, calculate_accuracy=False)
		examples = []
		for index in indices: 
			anchor = index[0]
			neighbors = index[1:]
			for neighbor in neighbors:
				examples.append((anchor, neighbor))
		df = pd.DataFrame(examples, columns=["anchor", "neighbor"])
		return df


	def get_predictions(self, model, dataloader):
		predictions, probs = [], []
		epoch_iterator = tqdm(dataloader, total=len(dataloader))
		model.eval()
		with torch.no_grad():
			for i, batch in enumerate(epoch_iterator):
				output_i = model(batch["anchor"])
				probs.extend(torch.nn.functional.softmax(output_i, dim=1).cpu().numpy())
				predictions.extend(torch.argmax(output_i, dim=1).cpu().numpy())
		return predictions, probs


	def train(self, model, optimizer, criterion, train_dataloader):
		train_iterator = range(int(self.args.num_epochs))
		# train
		for epoch in train_iterator:
			bar_desc = "Epoch %d of %d | Iteration" % (epoch + 1, len(train_iterator))
			epoch_iterator = tqdm(train_dataloader, desc=bar_desc)
			for step, batch in enumerate(epoch_iterator):
				optimizer.zero_grad()
				model.zero_grad()
				anchor, neighbor = batch["anchor"], batch["neighbor"]
				anchors_output, neighbors_output = model(anchor), model(neighbor)
				total_loss, consistency_loss, entropy_loss = criterion(anchors_output, neighbors_output)
				total_loss.backward()
				optimizer.step()
		optimizer.zero_grad()
		model.zero_grad()
		return model


	def plot_elbow(self, distortion_scores):
		locator_kwargs = {
		    "curve_nature": "convex",
		    "curve_direction": "decreasing",
		}

		try:
			range_n_clusters = list(range(self.args.min_clusters, self.args.max_clusters, self.args.stepsize))
			#print (range_n_clusters, distortion_scores)
			#print (len(range_n_clusters), len(distortion_scores))
			elbow_locator = KneeLocator(range_n_clusters, distortion_scores, **locator_kwargs)
			elbow_value_ = elbow_locator.knee
			elbow_score_ = range_n_clusters[range_n_clusters.index(elbow_value_)]
			elbow_label = "elbow at $k={}$".format(elbow_value_)

			plt.plot(range_n_clusters, distortion_scores)

			plt.axvline(elbow_value_, c="tab:red", linestyle="--", label=elbow_label)
			plt.legend()
			plt.savefig(os.path.join(self.args.outpath, "optimal_number_of_clusters.png"))
			return elbow_value_
		except Exception as e:
			print (str(e))
			print ("no knee found, perhaps something wrong with the data or the range of numbers of clusters searched over")
			return self.args.min_clusters

	def calculate_distortion_scores_and_train_model(self):
		# 	def __init__(self, neigbhors_df, embeddings, test_embeddings="", mode="train"):
		train_dataset = DocScanDataset(self.neighbor_dataset, self.X, mode="train")
		predict_dataset = DocScanDataset(self.neighbor_dataset, self.X, self.X, mode="predict")
		distortion_scores = []
		# iterate over specified amount of clusters to find optimal value
		for num_classes in range(self.args.min_clusters, self.args.max_clusters, self.args.stepsize):
			# initalize docscan model
			model = DocScanModel(num_classes, self.args.dropout).to(self.device)
			optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-8)
			criterion = SCANLoss()
			criterion.to(self.device)

			# get dataloaders
			batch_size = max(self.args.batch_size, self.args.num_classes * 4) # well, if we try to fit 300 clusters, we probably want a batchsize bigger than 64
			train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn = train_dataset.collate_fn, batch_size=batch_size)
			predict_dataloader = torch.utils.data.DataLoader(predict_dataset, shuffle=False, collate_fn = predict_dataset.collate_fn_predict, batch_size=batch_size)
			# train
			model = self.train(model, optimizer, criterion, train_dataloader)
			# predict
			predictions, probabilities = self.get_predictions(model, predict_dataloader)

			predictions = np.array(predictions)
			distortion = 0
			try:
				for cluster in range(num_classes):
					mask = predictions == cluster
					instances = self.embeddings[mask]
					center = instances.mean(axis=0)
					center = np.array([center])
					distances = pairwise_distances(instances, center, metric="euclidean")
					distances = distances ** 2
					# Add the sum of square distance to the distortion
					distortion += distances.sum()
				distortion_scores.append(distortion)
			except Exception as e:
				#print (str(e))
				#print ("failing with distortion scores, why? nclusters=", num_classes)
				#input("")
				pass

		#print ("distortion_scores", distortion_scores)
		#input("")
		# plot elbow
		elbow_value_ = self.plot_elbow(distortion_scores)
		# train model with optimal value
		num_classes = elbow_value_
		model = DocScanModel(num_classes, self.args.dropout).to(self.device)
		optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-8)
		criterion = SCANLoss()
		criterion.to(self.device)

		# get dataloaders
		batch_size = max(self.args.batch_size, self.args.num_classes * 4) # well, if we try to fit 300 clusters, we probably want a batchsize bigger than 64
		train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn = train_dataset.collate_fn, batch_size=batch_size)
		predict_dataloader = torch.utils.data.DataLoader(predict_dataset, shuffle=False, collate_fn = predict_dataset.collate_fn_predict, batch_size=batch_size)
		# train
		model = self.train(model, optimizer, criterion, train_dataloader)
		# predict
		predictions, probabilities = self.get_predictions(model, predict_dataloader)

		return predictions, probabilities, elbow_value_



	def write_prototypical_examples(self, df, outfile="prototypical_examples_by_clusters.txt"):
		with open(os.path.join(self.args.outpath, outfile), "w") as outfile:
			for topic in tqdm(np.unique(df["clusters"])):
				df_topic = df[df["clusters"] == topic.item()]
				probabilites = [i[int(topic)] for i in df["probabilities"]]
				indices = np.argsort(probabilites)[::-1][:10]
				for i in indices:
					outfile.write(str(topic) + "\t" + df.iloc[i]["sentence"] + "\n")

	def draw_wordclouds(self, df, outpath="wordclouds"):

		nlp = English()
		tokenizer = nlp.Defaults.create_tokenizer(nlp)
		nlp.add_pipe(nlp.create_pipe('sentencizer'))

		outpath = os.path.join(self.args.outpath, outpath)
		os.makedirs(outpath, exist_ok=True)
		if self.args.wordcloud_frequencies == "tf-idf":
			vectorizer = TfidfVectorizer(stop_words='english', min_df=5, max_df=0.75, max_features=10000)
			vectorizer.fit(df["sentence"])
		for topic in tqdm(np.unique(df["clusters"])):
			df_topic = df[df["clusters"] == topic.item()]
			if self.args.wordcloud_frequencies == "tf-idf":
				generate_word_clouds(topic, df_topic, nlp, outpath, vectorizer)
			else:
				generate_word_clouds(topic, df_topic, nlp, outpath)




	def run_main(self):
		# embedd using SBERT
		print ("loading data...")
		df = self.load_data()
		print ("embedding sentences...")
		self.embeddings = self.embedd_sentences(df["sentence"])

		# torch tensor of embeddings
		self.X = torch.from_numpy(self.embeddings)

		# generate neighbor dataset
		print ("retrieving neighbors...")
		self.memory_bank = MemoryBank(self.X, "", len(self.X), 
		                        self.X.shape[-1],
		                        self.args.num_classes)
		self.neighbor_dataset = self.create_neighbor_dataset()


		print ("compute optimal amount of clusters...")
		predictions, probabilities, elbow_value_ = self.calculate_distortion_scores_and_train_model()


		print ("docscan trained with n=", elbow_value_, "clusters...")
 
		df["clusters"] = predictions
		df["probabilities"] = probabilities

		# save docscan output
		df.to_csv(os.path.join(self.args.outpath, "docscan_clusters.csv"))

		# visualizations
		print ("finding prototypical sentences for each cluster...")
		self.write_prototypical_examples(df)

		# draw wordclouds
		print ("drawing wordclouds...")
		self.draw_wordclouds(df)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--infile", type=str, help="path to infile/dataframe")
	parser.add_argument("--data_format", type=str, default="from_txt", help="whether infile points to a txt file where each line is a sentence or to a dataframe")
	parser.add_argument("--outpath", type=str, help="path to output path where output of docscan gets saved")
	parser.add_argument("--sbert_model", default="bert-base-nli-mean-tokens", type=str, help="SBERT model to use to embedd sentences")
	parser.add_argument("--max_seq_length", default=128, type=int, help="max seq length of sbert model, sequences longer than this get truncated at this value")
	parser.add_argument("--topk", type=int, default=5, help="numbers of neighbors retrieved to build SCAN training set")
	parser.add_argument("--num_classes", type=int, default=10, help="numbers of clusters")
	parser.add_argument("--min_clusters", default=10, type=int, help="lower bound of clusters to search through to generate elbow")
	parser.add_argument("--max_clusters", default=22, type=int, help="upper bound of clusters to search through to generate elbow")
	parser.add_argument("--stepsize", default=2, type=int, help="2")
	parser.add_argument("--batch_size", default=64, type=int, help="Batch size per GPU/CPU for training.")
	parser.add_argument("--wordcloud_frequencies", default="tf-idf", type=str, help="wordcount weighting, either tf-idf or raw counts")
	parser.add_argument("--dropout", default=0.1, type=float, help="dropout for DocSCAN model")
	parser.add_argument("--num_epochs", default=3, type=int, help="number of epochs to train DocSCAN model")
	args = parser.parse_args()

	docscan = DocSCANPipeline(args)
	docscan.run_main()
