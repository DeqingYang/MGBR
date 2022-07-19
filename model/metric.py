import torch
import numpy as np

def compute_rr(inputs: torch.Tensor, ks:list) -> list:
    # print(inputs.shape)
    rr_k = [0 for _ in range(len(ks))]
    for j in range(len(ks)):
        if ks[j] > inputs.shape[1]:
            cutout = inputs
        else:
            cutout = inputs[:, 0:ks[j]]
        for i in range(cutout.shape[0]):
            test_sample = cutout[i]
            a = torch.argsort(test_sample, descending=True) # 返回的是排好序的索引
            b = torch.argsort(a)
            rank = b[0].item()
            if rank < ks[j]:
                rr_k[j] += 1 / (rank + 1)
    return rr_k


def compute_recall(inputs: torch.Tensor, ks: list) -> list:
    recall_k = [0 for i in range(len(ks))]
    for i in range(inputs.shape[0]):
        test_sample = inputs[i]
        a = torch.argsort(test_sample, descending=True)
        b = torch.argsort(a)
        rank = b[0].item()
        for j in range(len(ks)):
            if rank < ks[j]:
                recall_k[j] += 1
    return recall_k


def compute_ndcg(inputs: torch.Tensor, ks: list) -> list:
    ndcg_k = [0 for i in range(len(ks))]
    for j in range(len(ks)):
        if ks[j] > inputs.shape[1]:
            cutout = inputs
        else:
            cutout = inputs[:, 0:ks[j]]
        for i in range(cutout.shape[0]):
            test_sample = cutout[i]
            a = torch.argsort(test_sample, descending=True) # 返回的是排好序的索引
            b = torch.argsort(a)
            rank = b[0].item()
            if rank < ks[j]:
                ndcg_k[j] += 1 / np.log2(rank + 2)
    return ndcg_k
