conda create -n scan python=3.6
conda activate scan

pip install -U sentence-transformers
conda install faiss-cpu -c pytorch

pip install -r requirements.txt
