import pandas as pd
from sklearn.metrics import (
    r2_score,
    explained_variance_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    auc,
)
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from typing import Union, overload, Tuple, List
from tqdm.rich import tqdm
import numpy as np
# import confidenceinterval as ci
import ppp_prediction.metrics.confidenceinterval as ci
from statsmodels.stats.multitest import multipletests
import numpy as np
import statsmodels.stats.multitest as smm
from pandas import DataFrame
from itertools import product
import statsmodels.api as sm
from typing import Union, overload, Tuple, List
from sklearn.metrics import (
    r2_score,
    explained_variance_score,
    roc_auc_score,
    accuracy_score,
)
from scipy.stats import pearsonr, spearmanr
import scipy.stats as ss
from .utils import find_best_cutoff
from .ci import bootstrap_ci
from sklearn.metrics import confusion_matrix
from ppp_prediction.metrics.ci import bootstrap_ci
from sklearn.metrics import roc_curve
import numpy as np


def cal_binary_metrics(y, y_pred, ci=False, n_resamples=100):
    """Calculate common binary classification metrics.

    Args:
        y (array-like): True binary labels
        y_pred (array-like): Predicted probabilities or scores
        ci (bool, optional): Whether to calculate confidence intervals. Defaults to False.
        n_resamples (int, optional): Number of bootstrap resamples for CI. Defaults to 100.

    Returns:
        dict: Dictionary containing the following metrics:
            - AUC: Area under ROC curve
            - ACC: Accuracy score
            - Macro_F1: Macro-averaged F1 score
            - Sensitivity: True positive rate at optimal threshold
            - Specificity: True negative rate at optimal threshold
            - APR: Area under precision-recall curve
            If ci=True, returns bootstrap confidence intervals for these metrics.
    """

    if not ci:
        fpr, tpr, thresholds = roc_curve(y, y_pred)
        AUC = roc_auc_score(y, y_pred)
        # by best youden
        optim_threshold, optim_fpr, optim_tpr = find_best_cutoff(fpr, tpr, thresholds)
        y_pred_binary = (y_pred > optim_threshold).astype(int)
        ACC = accuracy_score(y, y_pred_binary)
        macro_f1 = f1_score(y, y_pred_binary, average="macro")
        sensitivity = optim_tpr
        specificity = 1 - optim_fpr
        precision, recall, _ = precision_recall_curve(y, y_pred)
        APR = auc(recall, precision)

        return {
            "AUC": AUC,
            "ACC": ACC,
            "Macro_F1": macro_f1,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "APR": APR,
        }
    elif ci:
        return cal_binary_metrics_bootstrap(
            y, y_pred, ci_kwargs={"n_resamples": n_resamples}
        )

def cal_qt_metrics(y_true, y_pred, ci=False, n_resamples=100):
    """
    Calculate quantitative metrics for model evaluation.

    This function computes various correlation and regression metrics between true and predicted values.

    Args:
        y_true (array-like): Ground truth (correct) target values
        y_pred (array-like): Estimated target values
        ci (bool, optional): Whether to calculate confidence intervals using bootstrap. Defaults to False
        n_resamples (int, optional): Number of bootstrap resamples for CI calculation. Defaults to 100

    Returns:
        dict: Dictionary containing the following metrics:
            - pearsonr: Pearson correlation coefficient
            - spearmanr: Spearman rank correlation coefficient
            - explained_variance_score: Explained variance regression score
            - r2_score: R^2 (coefficient of determination) regression score
            If ci=True, returns bootstrap confidence intervals for these metrics
    """

    if not ci:
        pearsonr_score = pearsonr(y_true, y_pred)[0]
        spearmanr_score = spearmanr(y_true, y_pred)[0]
        explained_variance_score_ = explained_variance_score(y_true, y_pred)
        r2_score_ = r2_score(y_true, y_pred)
        return {
            "pearsonr": pearsonr_score,
            "spearmanr": spearmanr_score,
            "explained_variance_score": explained_variance_score_,
            "r2_score": r2_score_,
        }
    elif ci:
        return cal_qt_metrics_bootstrap(
            y_true, y_pred, ci_kwargs={"n_resamples": n_resamples}
        )

def cal_qt_metrics_bootstrap(y_true, y_pred, ci_kwargs=None):
    """
    ci_kwargs:
        confidence_level=0.95
        method="bootstrap_bca"
        n_resamples=5000
    """

    ci_params = dict(confidence_level=0.95, method="bootstrap_basic", n_resamples=5000)
    if ci_kwargs is not None:
        ci_params.update(ci_kwargs)

    r2, (r2_LCI, r2_UCI) = bootstrap_ci(
        y_true=y_true,
        y_pred=y_pred,
        metric=lambda y_true, y_pred: r2_score(y_true, y_pred),
    )
    pearsonr_score, (pearsonr_LCI, pearsonr_UCI) = bootstrap_ci(
        y_true=y_true,
        y_pred=y_pred,
        metric=lambda y_true, y_pred: pearsonr(y_true, y_pred)[0],
    )



    return {
        "pearsonr": pearsonr_score,
        "pearsonr_LCI": pearsonr_LCI,
        "pearsonr_UCI": pearsonr_UCI,
        "r2_score": r2,
        "r2_score_LCI": r2_LCI,
        "r2_score_UCI": r2_UCI,
        "N": len(y_true),
    }


def cal_binary_metrics_bootstrap(y, y_pred, ci_kwargs=None):
    """
    Calculate binary classification metrics with bootstrap confidence intervals.

    Args:
        y (array-like): True binary labels
        y_pred (array-like): Predicted probabilities or scores
        ci_kwargs (dict, optional): Parameters for confidence interval calculation.
            Supported keys:
            - confidence_level (float): Confidence level (default: 0.95)
            - method (str): Bootstrap method (default: "bootstrap_basic")
            - n_resamples (int): Number of bootstrap resamples (default: 5000)

    Returns:
        dict: Dictionary containing the following metrics with confidence intervals:
            - AUC: Area under ROC curve
            - ACC: Accuracy
            - Macro_F1: Macro F1 score
            - Sensitivity: True positive rate
            - Specificity: True negative rate
            - APR: Average precision
            - N: Total sample size
            - N_case: Number of positive cases
            - N_control: Number of negative cases
            Each metric includes point estimate and upper/lower confidence bounds
    """

    ci_params = dict(confidence_level=0.95, method="bootstrap_basic", n_resamples=5000)
    if ci_kwargs is not None:
        ci_params.update(ci_kwargs)
    # by best youden
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    optim_threshold, optim_fpr, optim_tpr = find_best_cutoff(fpr, tpr, thresholds)
    y_pred_binary = (y_pred > optim_threshold).astype(int)

    # cal metrics
    AUC, AUC_CI = ci.roc_auc_score(y, y_pred, **ci_params)
    ACC, ACC_CI = ci.accuracy_score(y, y_pred_binary, **ci_params)
    macro_f1, macro_f1_CI = ci.f1_score(y, y_pred_binary, average="macro", **ci_params)
    sensitivity, sensitivity_CI = ci.tpr_score(y, y_pred_binary, **ci_params)  # TPR
    specificity, specificity_CI = ci.tnr_score(y, y_pred_binary, **ci_params)  # TNR

    APR, APR_CI = APR_bootstrap(y, y_pred, **ci_params)
    return {
        "AUC": AUC,
        "AUC_UCI": AUC_CI[1],
        "AUC_LCI": AUC_CI[0],
        "ACC": ACC,
        "ACC_UCI": ACC_CI[1],
        "ACC_LCI": ACC_CI[0],
        "Macro_F1": macro_f1,
        "Macro_F1_UCI": macro_f1_CI[1],
        "Macro_F1_LCI": macro_f1_CI[0],
        "Sensitivity": sensitivity,
        "Sensitivity_UCI": sensitivity_CI[1],
        "Sensitivity_LCI": sensitivity_CI[0],
        "Specificity": specificity,
        "Specificity_UCI": specificity_CI[1],
        "Specificity_LCI": specificity_CI[0],
        "APR": APR,
        "APR_UCI": APR_CI[1],
        "APR_LCI": APR_CI[0],
        "N": len(y),
        "N_case": y.sum(),
        "N_control": len(y) - y.sum(),
    }


def APR_score(y_true, y_pred):
    """
    Calculate the Area under the Precision-Recall curve (APR) score.

    Args:
        y_true (array-like): True binary labels.
        y_pred (array-like): Target scores, probability estimates or confidence values.

    Returns:
        float: Area under the precision-recall curve.
    """

    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    apr = auc(x=recall, y=precision)

    return apr


def APR_bootstrap(y_true, y_pred, **args):

    ci_params = dict(confidence_level=0.95, method="bootstrap_bca", n_resamples=5000)
    if args is not None:
        ci_params.update(args)

    APR, APR_CI = bootstrap_ci(
        y_true=y_true,
        y_pred=y_pred,
        metric=lambda y_true, y_pred: APR_score(y_true, y_pred),
        **ci_params,

    )
    return APR, APR_CI


def cal_DR(y_true, y_pred, ci=False, n_resamples=100):
    """
    Calculate Detection Rate (DR) from true and predicted labels.

    DR = TP / (FN + TP)

    Args:
        y_true: Array-like of true labels
        y_pred: Array-like of predicted labels
        ci: If True, calculate confidence intervals using bootstrap
        n_resamples: Number of bootstrap resamples for CI calculation

    Returns:
        If ci=False:
            float: Detection Rate
        If ci=True:
            tuple: (DR, DR_LCI, DR_UCI) containing Detection Rate and confidence intervals
    """

    if not ci:
        cm = confusion_matrix(y_true, y_pred)
        DR = cm[1, 1] / (cm[1, 0] + cm[1, 1])
        return DR
    else:
        DR, (DR_LCI, DR_UCI) = bootstrap_ci(
            y_true=y_true,
            y_pred=y_pred,
            metric=lambda y_true, y_pred: cal_DR(y_true, y_pred, ci=False),
            n_resamples=n_resamples,
        )
        return DR, DR_LCI, DR_UCI


def cal_DR_at_FPR(y_true, y_pred, at_min_fpr, ci=False, n_resamples=100):
    """Calculate detection rate (DR) at a given false positive rate (FPR) threshold.

    Args:
        y_true: Array of true binary labels
        y_pred: Array of predicted probabilities
        at_min_fpr: Target false positive rate threshold
        ci: If True, calculate confidence intervals using bootstrap resampling
        n_resamples: Number of bootstrap resamples for CI calculation

    Returns:
        Detection rate (float) or tuple of (DR, lower CI, upper CI) if ci=True
    """

    # get cutoff by fpr
    fpr, tpr, cutoff = get_cutoff_at_FPR(y_true, y_pred, at_min_fpr)

    y_pred = (y_pred > cutoff).astype(int)

    return cal_DR(y_true, y_pred, ci=ci, n_resamples=n_resamples)


def get_cutoff_at_FPR(
    y_true,
    y_pred,
    at_min_fpr,
):
    """
    y_pred should be continuous
    """
    # get cutoff by fpr
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # check the closest fpr to at_min_fpr
    idx = np.argmin(np.abs(fpr - at_min_fpr))
    cutoff = thresholds[idx]

    print(
        f"Given min_fpr: {at_min_fpr}, the closest fpr is {fpr[idx]:.2f} and cutoff is {cutoff:.2f} with tpr: {tpr[idx]:.2f}"
    )

    return tpr, fpr, cutoff


def cal_LR(
    y_true,
    y_pred,
    ci=False,
    n_resamples=100,
):
    if not ci:
        cm = confusion_matrix(y_true, y_pred)
        DR = cm[1, 1] / (cm[1, 0] + cm[1, 1])
        FPR = cm[0, 1] / (cm[0, 0] + cm[0, 1])
        LR = DR / FPR

        return LR
    else:

        LR, (LR_LCI, LR_UCI) = bootstrap_ci(
            y_true=y_true,
            y_pred=y_pred,
            metric=lambda y_true, y_pred: cal_LR(y_true, y_pred, ci=False),
            n_resamples=n_resamples,
        )
        return LR, LR_LCI, LR_UCI


def cal_LR_at_FPR(
    y_true,
    y_pred,
    at_min_fpr,
    ci=False,
    n_resamples=100,
):
    """
    y_pred should be continuous
    """
    # get cutoff by fpr
    fpr, tpr, cutoff = get_cutoff_at_FPR(y_true, y_pred, at_min_fpr)

    y_pred = (y_pred > cutoff).astype(int)

    return cal_LR(y_true, y_pred, ci=ci, n_resamples=n_resamples)
