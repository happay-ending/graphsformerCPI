import numpy as np
from math import sqrt

import torch
from sklearn import metrics
from sklearn.metrics import average_precision_score, mean_squared_error, r2_score, roc_auc_score, accuracy_score, \
    precision_recall_curve, precision_score, recall_score, f1_score
from scipy import stats


def get_aupr(Y, P, threshold=7.0):
    # print(Y.shape,P.shape)
    Y = np.where(Y >= 7.0, 1, 0)
    P = np.where(P >= 7.0, 1, 0)
    aupr = average_precision_score(Y, P)
    return aupr


def get_cindex(Y, P):
    summ = 0
    pair = 0

    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if (Y[i] > Y[j]):
                    pair += 1
                    summ += 1 * (P[i] > P[j]) + 0.5 * (P[i] == P[j])

    if pair is not 0:
        return summ / pair
    else:
        return 0


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


def get_rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse


def get_mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse


def get_pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def get_spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def get_ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci

def regression_scores(label, pred):
    label = np.array(label)
    pred = np.array(pred)
    # label = label.reshape(-1)
    # pred = pred.reshape(-1)
    rmse = sqrt(((label - pred) ** 2).mean(axis=0))
    mse = ((label - pred) ** 2).mean(axis=0)
    # pearson = np.corrcoef(label, pred)[0, 1]
    # print("label:",label)
    # print("pred:", pred)
    pearson = stats.pearsonr(label, pred)[0]
    spearman = stats.spearmanr(label, pred, nan_policy='omit')[0]
    ci = get_cindex(label, pred)
    d = {'rmse': rmse, 'mse': mse, 'pearson': pearson, 'spearman': spearman, 'ci': ci}
    return d


def classified_scores(y_true, y_score, y_pred=None, threshod=0.5):
    auc = roc_auc_score(y_true, y_score)
    if y_pred is None: y_pred = (y_score >= threshod).astype(int)
    acc = accuracy_score(y_true, y_pred)
    # tpr, fpr, _ = precision_recall_curve(y_true, y_score)
    # prauc = metrics.auc(fpr, tpr)
    # precision = precision_score(y_true, y_pred, average='macro')
    # recall = recall_score(y_true, y_pred, average='macro')
    # f1 = f1_score(y_true, y_pred, average='macro')
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    d = {'auc': auc, 'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1}
    return d



def cal_ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci


# def regression_metrics(y_true, y_pred):
#     # mae = mean_absolute_error(y_true, y_pred)
#     ci = cal_ci(y_true, y_pred)
#     mse = mean_squared_error(y_true, y_pred)
#     rmse = mse ** 0.5
#     pearson = stats.pearsonr(y_true, y_pred)[0]
#     spearman = stats.spearmanr(y_true, y_pred, nan_policy='omit')[0]
#     r2 = r2_score(y_true, y_pred)
#     # d = {'ci': ci, 'mse': mse, 'rmse': rmse, 'r2': r2}
#     # return d
#     return rmse, mse, pearson, spearman, ci


if __name__ == '__main__':
    from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryPrecisionRecallCurve, BinaryPrecision, \
        BinaryRecall, BinaryF1Score

    #    d = {'auc': auc, 'prauc': prauc, 'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1}
    preds = torch.tensor([0, 0, 1, 1, 0, 1], dtype=torch.float32)
    target = torch.tensor([0, 1, 1, 0, 0, 1], dtype=torch.float32)
    score = torch.tensor([0.11, 0.22, 0.84, 0.73, 0.33, 0.92], dtype=torch.float32)

    test_auc = BinaryAUROC(thresholds=None)
    test_prauc = BinaryPrecisionRecallCurve(thresholds=None)
    test_acc = BinaryAccuracy()
    test_precision = BinaryPrecision()
    test_recall = BinaryRecall()
    test_f1 = BinaryF1Score()

    auc = test_auc.update(score, target)
    acc = test_acc.update(preds, target)
    precision = test_precision(preds, target)
    recall = test_recall(preds, target)
    f1 = test_f1(preds, target)
    d = {'auc': auc, 'acc': acc.item(), 'precision': precision, 'recall': recall, 'f1': f1}
    print(d)

    preds = np.array([0, 0, 1, 1, 0, 1])
    target = np.array([0, 1, 1, 0, 0, 1])
    score = np.array([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])
    d_array = classified_scores(target, score, y_pred=preds)
    print(d_array)
