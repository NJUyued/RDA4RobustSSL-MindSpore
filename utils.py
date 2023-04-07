import torch
import torch.nn.functional as F
import numpy as np
import bisect
from kmeans_pytorch import kmeans
import mindspore as ms
from mindspore import nn
from mindspore import Tensor

from sklearn.cluster import KMeans

def linear_rampup(rampup_length):
    """Linear rampup"""
    def warpper(epoch):
        if epoch < rampup_length:
            return epoch / rampup_length
        else:
            return 1.0
    return warpper

def normalize_d(tensor):
    return tensor / ms.ops.reduce_sum(tensor)

def rda_consistency_loss(logits_w, logits_s, logits_w_A, logits_s_A, distri, distri_A, T=1.0, p_cutoff=0.0):
    logits_w = ms.ops.stop_gradient(logits_w)
    pseudo_label = ms.ops.softmax(logits_w)
    distri_A_ = Tensor(1, ms.float32) - distri_A
    distri_A_ = normalize_d(distri_A_)
    pseudo_label_da = normalize_d(pseudo_label * (ms.ops.reduce_mean(distri_A_) / ms.ops.reduce_mean(distri)))
    max_idx, max_probs = ms.ops.max(pseudo_label_da,axis=-1)

    logits_w_A = ms.ops.stop_gradient(logits_w_A)
    pseudo_label_A = ms.ops.softmax(logits_w_A)
    distri_ = Tensor(1, ms.float32) - distri
    distri_ = normalize_d(distri_)
    pseudo_label_A_da = normalize_d(pseudo_label_A * (ms.ops.reduce_mean(distri_) / ms.ops.reduce_mean(distri_A)))
    max_idx_A, max_probs_A = ms.ops.max(pseudo_label_A_da,axis=-1)

    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')
    loss = loss_fn(logits_s, max_idx) 
    loss_A = loss_fn(logits_s_A, max_idx_A) 

    return loss.mean() + loss_A.mean()

