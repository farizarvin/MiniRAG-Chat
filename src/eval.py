# src/eval.py
import math
import numpy as np

def precision_recall_f1(gold_set, retrieved_list):
    gold = set(gold_set)
    retrieved = list(retrieved_list)
    tp = len([d for d in retrieved if d in gold])
    precision = tp / len(retrieved) if retrieved else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1

def precision_at_k(retrieved, gold, k):
    topk = retrieved[:k]
    return sum(1 for d in topk if d in gold) / k

def average_precision(retrieved, gold, k):
    hits = 0.0
    s = 0.0
    for i, d in enumerate(retrieved[:k], start=1):
        if d in gold:
            hits += 1
            s += hits / i
    return s / len(gold) if gold else 0.0

def mean_average_precision(list_gold, list_retrieved, k):
    aps = []
    for gold, retrieved in zip(list_gold, list_retrieved):
        aps.append(average_precision(retrieved, gold, k))
    return np.mean(aps) if aps else 0.0

def dcg_at_k(relevances):
    return sum((rel / math.log2(idx + 2)) for idx, rel in enumerate(relevances))

def ndcg_at_k(retrieved, gold, k):
    rel = [1 if d in gold else 0 for d in retrieved[:k]]
    ideal = sorted(rel, reverse=True)
    idcg = dcg_at_k(ideal)
    if idcg == 0:
        return 0.0
    return dcg_at_k(rel) / idcg
