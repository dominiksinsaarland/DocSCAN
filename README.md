# DocSCAN

This is the code base for our paper [DocSCAN: Unsupervised Text Classification via Learning from Neighbors](https://aclanthology.org/2022.konvens-1.4/), accepted at [KONVENS 2022](https://konvens2022.uni-potsdam.de/).

## Installation

Assuming [Anaconda](https://docs.anaconda.com/anaconda/install/) and linux, the environment can be installed with the following command:
```shell
bash setup_environment.sh
```
## Run DOCSCAN on new dataset

First, one needs to create a folder with the data. We need two files in that folder: train.jsonl and test.jsonl

Both files contain lines where each line is a datapoint stored as a json dictionairy. The keys must be "text" and "label" (we don't need labels for train.jsonl, so this can contain dummies as well). 

We can then run the following script (first argument is the path to the folder, second argument the number of clusters). For the google_snippets dataset, we can run DOCScan with the following command

```shell
bash scripts/run_docscan.sh google_snippets 8
```
This generates a number of files in the folder:
* train_embedded.pkl (sbert embeddings for train file)
* test_embedded.pkl (sbert embeddings for test file)
* train_neighbours.jsonl (pairs of datapoint/neighbors for each datapoint)
* train_neighbours_embeddings.pkl (embeddings of pairs of datapoint/neighbors for each datapoint)
* mean_kmeans_tfidf.txt (averaged accuracy of 5 runs of kmeans with tf-idf features)
* mean_kmeans_sbert.txt (averaged accuracy of 5 runs of kmeans with sbert features)
* scan_results_classification_layer.txt (averaged accuracy of 5 runs of DOCScan)
* predictions.txt (SCAN predictions for the file test.jsonl)

## Run DOCSCAN on a raw txt file where each line is a datapoint

If we have a file where each line is an example, we can run docscan on this using the following script.

```shell
bash scripts/run_docscan_with_txt_file_as_input.sh $output_dir $num_clusters $path_to_filename
```

Where we have to specify $output_dir (which gets created in the script), the numbers of clusters and the path to the filename. It will then create a file "predictions.txt" in the output directory, one prediction per line where each line corresponds to the text example at this line in the input file.

## Replicate experiments in the paper

To replicate the experiments in the paper, we provide code below.

## Data for Experiments

to obtain the data used in the experiments in the paper, have a look at the file README_data.txt and download the required data

## Baselines

set up the data for experiments and compute embeddings by running:

```shell
bash scripts/setup_data.sh
```

to run the baselines (kmeans + svm baselines), run:

```shell
bash scripts/svm_baseline.sh
bash scripts/kmeans_baseline.sh
```

## Mining neighbors

Mine neighbours (raw text for RoBERTa experiments, and embeddings for only training classification layers) with:

```shell
bash scripts/find_nearest_neighbours
# to show accuracy for neighbors at k=1,5,50,100 for Figure 1, run e.g.
python src/find_nearest_neighbours.py --path google_snippets --show_accuracies
# training set accuracies [84.3, 80.4, 70.3, 65.8]
```

## Train SCAN classification layer

Train a classification layer with batchsize = 64, num_epochs = 3, dropout = 0.1, learning_rate = 0.001 on all datasets:


```shell
bash scripts/scan_classification_layer.sh 
# inspect results for e.g. google_snippets
cat google_snippets/scan_results_classification_layer.txt # each experiment is run 5 times and we show the mean and standard error for the 5 runs
```

## Train RoBERTa using SCAN

to train full RoBERTa on the neighbor datasets, run the following commands. Perhaps need to adjust batch_sizes and gradient accumulation steps if GPU out-of-memory errors (or switch to "--model_name distilbert-base-uncased" instead)
```shell
python src/scan_roberta.py --experiment 20newsgroup --batch_size 4 --gradient_accumulation_steps 8
python src/scan_roberta.py --experiment ag_news --batch_size 4 --gradient_accumulation_steps 8
python src/scan_roberta.py --experiment dbpedia --batch_size 4 --gradient_accumulation_steps 8
python src/scan_roberta.py --experiment google_snippets --batch_size 4 --gradient_accumulation_steps 8
python src/scan_roberta.py --experiment abstracts_group --batch_size 4 --gradient_accumulation_steps 8
```


