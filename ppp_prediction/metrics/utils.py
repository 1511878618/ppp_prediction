import numpy as np
def find_best_cutoff(fpr, tpr, thresholds):
    diff = tpr - fpr
    Youden_index = np.argmax(diff)
    optimal_threshold = thresholds[Youden_index]
    optimal_FPR, optimal_TPR = fpr[Youden_index], tpr[Youden_index]
    return optimal_threshold, optimal_FPR, optimal_TPR


from scipy.stats import norm


def cal_pvalue(mean, se, uci=None, lci=None, z=None):
    """

    SE = (UCI - LCI)/3.92
    """
    if uci and lci:
        se = (uci - lci) / 3.92
    if not z:
        z = abs(mean / se)
    pvalue = norm.sf(abs(z)) * 2

    return pvalue