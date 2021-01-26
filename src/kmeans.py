from sklearn.cluster import MiniBatchKMeans
import pandas as pd
import numpy as np
import os
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
import argparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC

def evaluate_(y, preds):
	print(metrics.classification_report(y, preds))
	#print(metrics.confusion_matrix(y, preds))
	print("accuracy", metrics.accuracy_score(y, preds))
	return metrics.classification_report(y, preds)

def evaluate(predicted, true):
	mapping = defaultdict(lambda: defaultdict(int))
	for i,j in zip(predicted, true):
		mapping[i][j] += 1

	best_mapping = {}
	for class_, values in mapping.items():
		print (class_, values)
		predicted_class = sorted(values.items(), key=lambda x:x[1])[-1]
		print (predicted_class)
		predicted_class = predicted_class[0]
		best_mapping[class_] = predicted_class

	predicted_mapped = [best_mapping[i] for i in predicted]	
	return evaluate_(true, predicted_mapped)


def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = len(flat_targets)

    assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res

def hungarian_evaluate(targets, predictions, class_names=None, 
                        compute_purity=True, compute_confusion_matrix=False,
                        confusion_matrix_file=None):
    # Evaluate model based on hungarian matching between predicted cluster assignment and gt classes.
    # This is computed only for the passed subhead index.

    # Hungarian matching

    num_classes = len(np.unique(targets))
    num_elems = len(targets)
    match = _hungarian_match(predictions, targets, preds_k=num_classes, targets_k=num_classes)
    reordered_preds = np.zeros(num_elems, dtype=predictions.dtype)
    for pred_i, target_i in match:
        reordered_preds[predictions == int(pred_i)] = int(target_i)

    # Gather performance metrics
    acc = int((reordered_preds == targets).sum()) / float(num_elems)
    nmi = metrics.normalized_mutual_info_score(targets, predictions)
    ari = metrics.adjusted_rand_score(targets, predictions)
    
    return {'ACC': acc, 'ARI': ari, 'NMI': nmi, 'hungarian_match': match}

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--path", type=str, default="20_newsgroup_preprocessed", help="")
	parser.add_argument("--features", type=str, default="sbert", help="")
	parser.add_argument("--train_file", type=str, default="")
	parser.add_argument("--test_file", type=str, default="")


	args = parser.parse_args()

	if args.train_file and args.test_file:
		fn_train = args.train_file
		fn_test = args.test_file
	else:
		fn_train = os.path.join(args.path, "train_embedded.pkl")
		fn_test = os.path.join(args.path, "test_embedded.pkl")

	df_train = pd.read_pickle(fn_train)
	# shuffle
	df_train = df_train.sample(frac=1)
	df_test = pd.read_pickle(fn_test)
	print (args.features)

	if args.features == "sbert":
		X_train, X_test = np.array(df_train["embeddings"].tolist()), np.array(df_test["embeddings"].tolist())
	elif args.features == "tfidf":
		vect = CountVectorizer(max_df = 0.7 , min_df=3, ngram_range=(1,2), lowercase=True, stop_words="english")
		tfidf = TfidfTransformer()
		vectorized_train = vect.fit_transform(df_train["sentence"])
		X_train = tfidf.fit_transform(vectorized_train)
		vectorized_test = vect.transform(df_test["sentence"])
		X_test = tfidf.transform(vectorized_test)
	results = []
	n_clusters = len(np.unique(df_train["label"]))
	print (n_clusters)
	print (np.unique(df_test["label"], return_counts=True))
	for _ in range(5):
		kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=512).fit(X_train)
		labels = kmeans.predict(X_test)
		classification_report = evaluate(labels, df_test["label"])
		label_map = {j:i for i,j in enumerate(np.unique(df_test["label"]))}
		targets = [label_map[i] for i in df_test["label"]]
		class_names = np.unique(df_test["label"])
		hungarian_match_metrics = hungarian_evaluate(np.array(targets), np.array(labels), class_names=class_names)
		with open(os.path.join(args.path, "results_kmeans_" + args.features + ".txt"), "w") as outfile:
			outfile.write(classification_report + "\n")
			for i,j in hungarian_match_metrics.items():
				outfile.write(i + " " + str(j) + "\n")
		results.append(hungarian_match_metrics["ACC"])
	print (results)
	with open(os.path.join(args.path, "mean_kmeans_" + args.features + ".txt"), "w") as outfile:
		print (np.round(np.mean(results), 3), np.std(results))
		outfile.write("accuracy: " + str(np.round(np.mean(results), 3)) + "\nstandard error: " + str(np.round(np.std(results), 3)))




