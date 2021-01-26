"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch
from sklearn import metrics

def evaluate(y, preds):
	print(metrics.classification_report(y, preds))
	#print(metrics.confusion_matrix(y, preds))
	print("accuracy", metrics.accuracy_score(y, preds))


class MemoryBank(object):
    def __init__(self, features, targets, n, dim, num_classes):
        self.n = n
        self.dim = dim 
        #self.features = torch.FloatTensor(self.n, self.dim)
        #self.targets = torch.LongTensor(self.n)
        self.features = features
        self.targets = targets
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.C = num_classes


    def mine_nearest_neighbors(self, topk, calculate_accuracy=True, show_eval=False):
        # mine the topk nearest neighbors for every sample
        import faiss
        features = self.features.cpu().numpy()
        n, dim = features.shape[0], features.shape[1]
        index = faiss.IndexFlatIP(dim)
        #index = faiss.index_cpu_to_all_gpus(index)
        index.add(features)
        distances, indices = index.search(features, topk+1) # Sample itself is included
        
        # evaluate 
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            neighbor_targets = np.take(targets, indices[:,1:], axis=0) # Exclude sample itself for eval
            anchor_targets = np.repeat(targets.reshape(-1,1), topk, axis=1)
            accuracy = np.mean(neighbor_targets == anchor_targets)
            if show_eval:
                print (np.shape(neighbor_targets), np.shape(anchor_targets))
                evaluate(anchor_targets.flatten(), neighbor_targets.flatten())  
            return indices, accuracy
        
        else:
            return indices

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')
