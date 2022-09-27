# DocSCAN

This is the code base for our paper [DocSCAN: Unsupervised Text Classification via Learning from Neighbors](https://aclanthology.org/2022.konvens-1.4/), accepted at [KONVENS 2022](https://konvens2022.uni-potsdam.de/).

## Update 27.09.2022

Major code refactoring, the whole repo should be way more user friendly now!

## Installation

Assuming [Anaconda](https://docs.anaconda.com/anaconda/install/) and linux, the environment can be installed with the following command:
```shell
conda create -n scan python=3.6
conda activate scan

pip install -U sentence-transformers
conda install faiss-cpu -c pytorch
pip install -r requirements.txt
```
## Run DOCSCAN on text dataset

Run with

```shell
PYTHONPATH=src python src/DocSCAN.py --infile 20newsgroup_sample.txt --outpath 20newsgroup --num_classes 20 
```

Where --infile is a file with one sentence per line, outpath is the output directory and we provide the number of clusters. The main output is a csv file in the output directory called docscan_clusters.csv with columns sentence, clusters and probabilities. 

Other output generated is
* prototypical_examples_by_clusters.txt (the 10 most likely sentences for each cluster)
* word clouds for each cluster in folder outpath/wordclouds

Also, if no number of classes is provided, the program automatically determines the number of clusters using an adapted elbow method from [yellowbrick](https://www.scikit-yb.org/en/latest/api/cluster/elbow.html). If so, --min_clusters, --max_clusters and --stepsize should be provided. However, this is rather experimental! 

### other input format

input can also be a pandas dataframe with a column "sentence". If this is more convenient, run with

```shell
PYTHONPATH=src python src/DocSCAN.py --infile 20newsgroup_sample.csv --outpath 20newsgroup --num_classes 20 --data_format from_csv
```


## Replicate Paper Experiments

Run with 

```shell
PYTHONPATH=src python src/DocSCAN_paper_replication.py --path 20_newsgroup 
```

path needs to contain 2 files, train.jsonl and test.jsonl where each line is a json dictionary containing the keys "text" and "label". Have a look at 
scripts/setup_data.sh for how these files were created.

## kmeans Baseline Experiments

Run with 

```shell
PYTHONPATH=src python src/kmeans.py --path 20_newsgroup 
```

## questions

If anything should not work or is unclear, please don't hesitate to contact the authors

* Dominik Stammbach (dominik.stammbach@gess.ethz.ch)

