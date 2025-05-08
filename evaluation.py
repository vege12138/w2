from sklearn import metrics
from munkres import Munkres
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib
import os
matplotlib.use('Agg')



def cluster_evaluation(pred, labels):
    cluster_num = len(torch.unique(labels))
    l1 = l2 = range(cluster_num)
    cost = np.zeros((cluster_num, cluster_num), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(labels) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(pred) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(labels, new_predict)
    f1_macro = metrics.f1_score(labels, new_predict, average='macro')
    nmi = metrics.normalized_mutual_info_score(labels, pred, average_method='geometric')
    return acc, nmi, f1_macro