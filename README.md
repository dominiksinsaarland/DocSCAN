# SCAN_for_NLP
Learning from Neighbors: Unsupervised Text Classification


## Installation

Assuming [Anaconda](https://docs.anaconda.com/anaconda/install/) and linux, the environment can be installed with the following command:
```shell
bash setup_environment.sh
```

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


