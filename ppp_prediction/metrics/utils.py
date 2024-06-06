import numpy as np
def find_best_cutoff(fpr, tpr, thresholds):
    diff = tpr - fpr
    Youden_index = np.argmax(diff)
    optimal_threshold = thresholds[Youden_index]
    optimal_FPR, optimal_TPR = fpr[Youden_index], tpr[Youden_index]
    return optimal_threshold, optimal_FPR, optimal_TPR


from scipy.stats import norm
import pandas as pd 

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


def format_metrics(x, x_uci, x_lci, data=None, round=2):
    if isinstance(data, pd.DataFrame):
        if isinstance(x, str):
            x = data[x]
        if isinstance(x_uci, str):
            x_uci = data[x_uci]
        if isinstance(x_lci, str):
            x_lci = data[x_lci]

    if isinstance(x, pd.Series):
        return (
            x.apply(lambda x: f"{x:.{round}f}").astype(str)
            + " ("
            + x_lci.apply(lambda x: f"{x:.{round}f}").astype(str)
            + ", "
            + x_uci.apply(lambda x: f"{x:.{round}f}").astype(str)
            + ")"
        )
    else:
        return x + " (" + f"{x_lci:.{round}f}" + ", " + f"{x_uci:.{round}f}" + ")"