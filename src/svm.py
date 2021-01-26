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
from sklearn.linear_model import SGDClassifier


def evaluate(y, preds):
	print(metrics.classification_report(y, preds))
	print("accuracy", metrics.accuracy_score(y, preds))
	return metrics.classification_report(y, preds), metrics.accuracy_score(y, preds)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--path", type=str, default="20_newsgroup_preprocessed", help="")
	parser.add_argument("--topk", type=int, default=5, help="")
	parser.add_argument("--features", type=str, default="sbert", help="")


	args = parser.parse_args()
	fn_train = os.path.join(args.path, "train_embedded.pkl")
	fn_test = os.path.join(args.path, "test_embedded.pkl")

	df_train = pd.read_pickle(fn_train)
	df_train = df_train.sample(frac=1)
	df_test = pd.read_pickle(fn_test)
	print (args.features)

	if args.features == "sbert":
		print (df_train)
		X_train, X_test = np.array(df_train["embeddings"].tolist()), np.array(df_test["embeddings"].tolist())
		print (np.shape(X_train), np.shape(X_test))
	elif args.features == "tfidf":
		vect = CountVectorizer(max_df = 0.7 , min_df=3, ngram_range=(1,2), lowercase=True, stop_words="english")
		tfidf = TfidfTransformer()
		vectorized_train = vect.fit_transform(df_train["sentence"])
		X_train = tfidf.fit_transform(vectorized_train)
		vectorized_test = vect.transform(df_test["sentence"])
		X_test = tfidf.transform(vectorized_test)

	svm = SGDClassifier(loss='hinge', penalty='l2',
                           alpha=1e-5, random_state=42,
                           max_iter=500, tol=1e-5, class_weight=None, verbose=1)
	
	svm.fit(X_train, np.array(df_train["label"]))
	labels = svm.predict(X_test)
	classification_report, acc = evaluate(labels, df_test["label"])

	with open(os.path.join(args.path, "results_svm_" + args.features + ".txt"), "w") as outfile:
		outfile.write(classification_report + "\n")
		outfile.write("ACCURACY" + str(acc))






