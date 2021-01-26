import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics

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
    #print (num_correct)
    #print (num_samples - num_correct)
    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    #print (match)
    match = np.array(list(zip(*match)))
    #print (match)
    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))
    return res

def hungarian_evaluate(targets, predictions, class_names=None, 
                        compute_purity=True, compute_confusion_matrix=True,
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

    classification_report = metrics.classification_report(targets, reordered_preds)
    cm = metrics.confusion_matrix(targets, reordered_preds)
    
    #_, preds_top5 = probs.topk(5, 1, largest=True)
    #reordered_preds_top5 = torch.zeros_like(preds_top5)
    #for pred_i, target_i in match:
    #    reordered_preds_top5[preds_top5 == int(pred_i)] = int(target_i)
    #correct_top5_binary = reordered_preds_top5.eq(targets.view(-1,1).expand_as(reordered_preds_top5))
    #top5 = float(correct_top5_binary.sum()) / float(num_elems)

    #return {'ACC': acc, 'ARI': ari, 'NMI': nmi, 'ACC Top-5': top5, 'hungarian_match': match}
    #confusion_matrix_file = "cm"
    #confusion_matrix(reordered_preds, targets, 
    #                class_names, confusion_matrix_file)

    return {'ACC': acc, 'ARI': ari, 'NMI': nmi, 'hungarian_match': match, "classification_report": classification_report, "confusion matrix": cm}

