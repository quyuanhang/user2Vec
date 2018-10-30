import math
import numpy as np

def evaluate(predictions, top_K):
    hits, ndcgs, aucs = [], [], []
    for scores in predictions:
        hr, ndcg, auc = eval_one_rating(scores, top_K)
        hits.append(hr)
        ndcgs.append(ndcg)
        aucs.append(auc)
    return np.array(hits).mean(), np.array(ndcgs).mean(), np.array(aucs).mean()

def eval_one_rating(scores, top_K):
    pos_score = scores[0]
    false_posi = scores[scores > pos_score]
    false_posi_strict = scores[scores >= pos_score]
    pos_rank = (len(false_posi) + len(false_posi_strict)) / 2
    # pos_rank = len(false_posi)
    hr = get_hit_ratio(pos_rank, top_K)
    # hr = pos_score
    ndcg = get_NDCG(pos_rank)
    # ndcg = scores.mean()
    auc = get_AUC(pos_rank, len(scores))
    return hr, ndcg, auc

def get_hit_ratio(pos_rank, top_K):
    return int(pos_rank < top_K)

def get_NDCG(pos_rank):
    return math.log(2) / math.log(pos_rank + 2)

def get_AUC(pos_rank, n):
    # print(pos_rank, n)
    auc = 1 - pos_rank / n
    return auc
