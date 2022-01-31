

from scipy.stats import stats, spearmanr
import numpy as np


# Calculate the BR@K, WR@K
def minmax_n_at_k(predict_scores, true_scores, ks=[0.001, 0.005]): # [0.001, 0.005, 0.01, 0.05, 0.10, 0.20]
    # print(ks)
    true_scores = np.array(true_scores)
    predict_scores = np.array(predict_scores)
    num_archs = len(true_scores)
    true_ranks = np.zeros(num_archs)
    true_ranks[np.argsort(true_scores)] = np.arange(num_archs)[::-1]
    predict_best_inds = np.argsort(predict_scores)[::-1]
    minn_at_ks = []
    for k in ks:
        ranks = true_ranks[predict_best_inds[:int(k * len(true_scores))]]
        if len(ranks) < 1:
            continue
        minn = int(np.min(ranks)) + 1
        maxn = int(np.max(ranks)) + 1
        minn_at_ks.append((k, minn, float(minn) / num_archs, maxn, float(maxn) / num_archs))
    return minn_at_ks


def get_best(scores, accs):
    accs = np.array(accs)
    scores = np.array(scores)
    num_archs = len(accs)
    true_ranks = np.zeros(num_archs)
    true_ranks[np.argsort(accs)] = np.arange(num_archs)[::-1]
    predict_best_inds = np.argsort(scores)[::-1] # score从大到小，idx坐标
    top_n = []

    # ranks = true_ranks[predict_best_inds]
    top1_rank_score = true_ranks[predict_best_inds[0]]
    top1_acc_score = accs[predict_best_inds[0]]

    return top1_rank_score, top1_acc_score


# Calculate ratio of top_ranking to the whole BR
def ratio_top_ranking_to_whole(predict_scores, true_scores):
    # # print(ks)
    # true_scores = np.array(true_scores)
    # predict_scores = np.array(predict_scores)
    # num_archs = len(true_scores)
    # true_ranks = np.zeros(num_archs)
    # true_ranks[np.argsort(true_scores)] = np.arange(num_archs)[::-1]
    # predict_best_inds = np.argsort(predict_scores)[::-1] # score从大到小，idx坐标
    
    # # ranks = true_ranks[predict_best_inds]
    # top1_rank_score = true_ranks[predict_best_inds[0]]
    # rank_top10_score = true_ranks[predict_best_inds[:10]]
    # # minn = int(np.min(ranks)) + 1
    # # maxn = int(np.max(ranks)) + 1

    top1_rank_score, _ = get_best(predict_scores, true_scores)
    top_n = []

    num_archs = len(true_scores)
    # return float(top1_rank_score) / num_archs
    top_n.append((1, float(top1_rank_score) / num_archs))
    # top_n.append((10, np.mean(rank_top10_score) / num_archs))
    return top_n


def mean_BR(predict_scores, true_scores, attributes, num_group=5):
    max_ = max(attributes)
    min_ = min(attributes)
    interval = (max_ - min_) / num_group
    attributes_group = [[] for i in range(num_group)]
    predict_scores_group = [[] for i in range(num_group)]
    true_scores_group = [[] for i in range(num_group)]
    for i, s in enumerate(predict_scores):
        idx = int((attributes[i] - min_) / interval)
        if idx >= num_group: idx = num_group - 1
        for j in range(idx, num_group): # small -> big
            attributes_group[j].append(attributes[i])
            predict_scores_group[j].append(s)
            true_scores_group[j].append(true_scores[i])
    
    BRs = []
    for i in range(num_group):
        BR = ratio_top_ranking_to_whole(predict_scores_group[i], true_scores_group[i])
        BRs.append(BR[0][-1])
        # BRs.append(BR)
        # BRs.append(BR[1][-1])
    
    BRs.append(sum(BRs)/num_group)
    return BRs


# Calculate the P@topK, P@bottomK, and Kendall-Tau in predicted topK/bottomK
def p_at_tb_k(predict_scores, true_scores, ratios=[0.01, 0.05]): # [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    # print(ratios)
    predict_scores = np.array(predict_scores)
    true_scores = np.array(true_scores)
    predict_inds = np.argsort(predict_scores)[::-1]
    num_archs = len(predict_scores)
    true_ranks = np.zeros(num_archs)
    true_ranks[np.argsort(true_scores)] = np.arange(num_archs)[::-1]
    patks = []
    for ratio in ratios:
        k = int(num_archs * ratio)
        if k < 1:
            continue
        top_inds = predict_inds[:k]
        bottom_inds = predict_inds[num_archs-k:]
        p_at_topk = len(np.where(true_ranks[top_inds] < k)[0]) / float(k)
        p_at_bottomk = len(np.where(true_ranks[bottom_inds] >= num_archs - k)[0]) / float(k)
        kd_at_topk = stats.kendalltau(predict_scores[top_inds], true_scores[top_inds]).correlation
        kd_at_bottomk = stats.kendalltau(predict_scores[bottom_inds], true_scores[bottom_inds]).correlation
        # [ratio, k, P@topK, P@bottomK, KT in predicted topK, KT in predicted bottomK]
        patks.append((ratio, k, p_at_topk, p_at_bottomk, kd_at_topk, kd_at_bottomk))
    return patks


criteria = {
    "LC": lambda x, y: np.corrcoef(x, y)[0][1],
    "KD": lambda x, y: stats.kendalltau(x, y, nan_policy='omit').correlation,
    "SpearmanR": lambda x, y: spearmanr(x, y, nan_policy='omit').correlation,
    'BR': lambda x, y: ratio_top_ranking_to_whole(x, y),
    'mBR': lambda x, y, attributes: mean_BR(x, y, attributes),
}