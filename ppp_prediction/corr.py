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
from tqdm import tqdm
import numpy as np
from confidenceinterval.bootstrap import bootstrap_ci
import confidenceinterval as ci
from statsmodels.stats.multitest import multipletests

import pandas as pd

import statsmodels.api as sm
from typing import Union, overload, Tuple, List
from sklearn.metrics import (
    r2_score,
    explained_variance_score,
    roc_auc_score,
    accuracy_score,
)
from scipy.stats import pearsonr, spearmanr
def generate_multipletests_result(df, pvalue_col='pvalue',alpha=0.05, method='fdr_bh'):
    df = df.copy() 
    pvalue_series = df[pvalue_col]
    reject, pvals_corrected, _, _ = multipletests(pvalue_series, alpha=alpha, method='fdr_bh')
    df['pval_corrected'] = pvals_corrected
    df['reject'] = reject
    return df 



def find_best_cutoff(fpr, tpr, thresholds):
    diff = tpr - fpr
    Youden_index = np.argmax(diff)
    optimal_threshold = thresholds[Youden_index]
    optimal_FPR, optimal_TPR = fpr[Youden_index], tpr[Youden_index]
    return optimal_threshold, optimal_FPR, optimal_TPR


def cal_binary_metrics(y, y_pred):
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


def APR_score(y_true, y_pred):
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
        metric=lambda y1, y2: APR_score(y1, y2),
        **ci_params,
    )
    return APR, APR_CI


def cal_binary_metrics_bootstrap(y, y_pred, ci_kwargs=None):
    """
    ci_kwargs:
        confidence_level=0.95
        method="bootstrap_bca"
        n_resamples=5000
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
    }


def split_list_into_k_chunks(lst, k):
    """
    将列表lst分成k个尽可能平均的子列表

    参数:
    lst: 要分割的原始列表
    k: 目标子列表的数量

    返回:
    一个列表，包含k个子列表（chunks）
    """
    # 计算每个chunk的平均长度，使用整除得到基本长度
    chunk_size = len(lst) // k
    # 计算需要额外分配一个元素的chunk数量
    chunks_with_extra = len(lst) % k

    chunks = []
    start_index = 0
    for i in range(k):
        # 如果当前chunk在前chunks_with_extra个chunk中，则长度为chunk_size+1
        # 否则长度为chunk_size
        end_index = start_index + chunk_size + (1 if i < chunks_with_extra else 0)
        # 从原始列表中切片，创建新的chunk
        chunk = lst[start_index:end_index]
        chunks.append(chunk)
        # 更新下一个chunk的起始索引
        start_index = end_index

    return chunks


def parallel_cal_corr(df_x_tuple, y, cofounders, model_type, family):
    df, x = df_x_tuple
    return cal_corr(df, x, y, cofounders, model_type, family, return_all=False)


@overload
def cal_corr(
    df,
    x: List[str],
    y: str,
    cofounders: Union[str, list] = None,
    model_type="glm",
    family=sm.families.Gaussian(),
) -> pd.DataFrame: ...
@overload
def cal_corr(
    df,
    x: str,
    y: str,
    cofounders: Union[str, list] = None,
    model_type="glm",
    family=sm.families.Gaussian(),
) -> pd.DataFrame: ...
@overload
def cal_corr(
    df,
    x: str,
    y: str,
    cofounders: Union[str, list] = None,
    model_type="ols",
    family=sm.families.Gaussian(),
    return_all=True,
) -> Tuple[sm.GLM, pd.DataFrame, pd.DataFrame]: ...
def cal_corr(
    df,
    x: Union[str, List[str]],
    y: str,
    cofounders: Union[str, list] = None,
    model_type="glm",
    family=sm.families.Gaussian(),
    return_all=False,
    parallel_cores=None,
    bootstrap_nums=0,
) -> pd.DataFrame:
    if isinstance(x, list):
        if parallel_cores is not None and return_all == False:
            from multiprocessing import Pool
            from functools import partial

            x_split = split_list_into_k_chunks(x, k=parallel_cores)
            df_x_split = [(df[x_ + cofounders + [y]], x_) for x_ in x_split]

            parallel_func = partial(
                parallel_cal_corr,
                y=y,
                cofounders=cofounders,
                model_type=model_type,
                family=family,
                bootstrap_nums=bootstrap_nums,
            )

            with Pool(parallel_cores) as p:
                res = list(
                    tqdm(p.imap(parallel_func, df_x_split), total=len(df_x_split))
                )

            return pd.concat(res)
        result = []
        for x_ in tqdm(x):
            current_res = cal_corr(
                df,
                x_,
                y,
                cofounders,
                model_type,
                family,
                return_all=return_all,
                bootstrap_nums=bootstrap_nums,
            )
            if current_res is not None:
                result.append(current_res)

        if return_all:
            model_list = [x[0] for x in result]
            result_df = [x[1] for x in result]
            pred_df = [x[2] for x in result]
            return model_list, pd.concat(result_df, axis=1).T, pred_df
        # return pd.concat(result, axis=1 ).T
        else:
            return pd.concat(result, axis=1).T
        # return result

    if isinstance(cofounders, str):
        cofounders = [cofounders]
    elif cofounders is None:
        cofounders = []

    assert model_type in [
        "glm",
        "ols",
        "logit",
    ], "model should be one of ['glm', 'ols', 'logit']"
    print(f"passed data have {df.shape[0]} rows")
    used_df = df[[x, y] + cofounders].dropna()

    X = sm.add_constant(used_df[[x] + cofounders])
    Y = used_df[y]
    print(f"used data have {used_df.shape[0]} rows after dropna")
    if model_type == "glm":
        model = sm.GLM(Y, X, family=family).fit()
        y_pred = model.predict(X)

        metrics = {"Persudo_R2": model.pseudo_rsquared()}
    elif model_type == "ols":
        model = sm.OLS(Y, X).fit()
        y_pred = model.predict(X)

        metrics = {"Persudo_R2": model.pseudo_rsquared()}
    elif model_type == "logit":
        model = sm.Logit(Y, X).fit()
        y_pred = model.predict(X)
        if bootstrap_nums > 0:
            metrics = cal_binary_metrics_bootstrap(
                Y, y_pred, ci_kwargs={"n_resamples": bootstrap_nums}
            )
        else:
            metrics = cal_binary_metrics(Y, y_pred)

    y_true = used_df[y]
    metrics.update(
        {
            "pearsonr": pearsonr(y_true, y_pred)[0],
            "spearmanr": spearmanr(y_true, y_pred)[0],
            "explained_variance_score": explained_variance_score(y_true, y_pred),
            "r2_score": r2_score(y_true, y_pred),
        }
    )

    model_res = (
        model.summary2()
        .tables[1]
        .rename(
            columns={
                "Coef.": "coef",
                "Std.Err.": "std",
                "z": "z",
                "P>|z|": "pvalue",
                "[0.025": "upper_ci",
                "0.975]": "lower_ci",
            }
        )
        .loc[x]
        .to_dict()
    )

    result = {
        "var": x,
        "exposure": y,
        "model": model_type,
        "pvalue": model.pvalues[x],
        "coef": model.params[x],
        "std": model_res["std"],
        "z": model_res["z"],
        "upper": model_res["upper_ci"],
        "lower": model_res["lower_ci"],
    }
    result.update(metrics)

    if len(used_df[y].unique()) <= 2:
        case_control_metrics = {
            "n_case": used_df[y].sum(),
            "n_control": used_df.shape[0] - used_df[y].sum(),
        }
        result.update(case_control_metrics)

    used_df[f"{y}_pred"] = y_pred

    result = pd.Series(result)
    if return_all:
        return model, result, used_df
    return result


import numpy as np
import statsmodels.stats.multitest as smm


def generate_states_cols(res_df, pvalue_col="pvalue"):
    res_df = res_df.copy()
    res_df["LOG10P"] = -np.log10(res_df[pvalue_col].astype(float))
    p_values = res_df[pvalue_col]

    # 使用statsmodels库中的multipletests函数计算q值
    reject, q_values, _, _ = smm.multipletests(p_values, method="fdr_bh")

    res_df["q_values"] = q_values
    res_df["reject"] = reject

    conditions = [
        (res_df["q_values"] < 1e-2) & (res_df["q_values"] >= 1e-3),
        (res_df["q_values"] < 1e-3) & (res_df["q_values"] >= 1e-4),
        (res_df["q_values"] < 1e-4),
    ]
    choices = ["*", "**", "***"]
    res_df["markers"] = np.select(conditions, choices, default="")
    return res_df



def cal_corr_multivar(
    df,
    x: Union[str, List[str]],
    y: str,
    cofounders: Union[str, list] = None,
    model_type="glm",
    family=sm.families.Gaussian(),
    return_all=False,
) -> pd.DataFrame:

    if isinstance(cofounders, str):
        cofounders = [cofounders]
    elif cofounders is None:
        cofounders = []

    assert model_type in [
        "glm",
        "ols",
        "logit",
    ], "model should be one of ['glm', 'ols', 'logit']"
    print(f"passed data have {df.shape[0]} rows")
    used_df = df[x + [y] + cofounders].dropna()

    X = sm.add_constant(used_df[x + cofounders])
    Y = used_df[y]
    print(f"used data have {used_df.shape[0]} rows after dropna")
    print(f"using input variables : {x + cofounders} to predict {y}")
    if model_type == "glm":
        model = sm.GLM(Y, X, family=family).fit()
        y_pred = model.predict(X)
        additional_metrics = {}
    elif model_type == "ols":
        model = sm.OLS(Y, X).fit()
        y_pred = model.predict(X)
        additional_metrics = {}
    elif model_type == "logit":
        model = sm.Logit(Y, X).fit()
        y_pred = model.predict(X)
        additional_metrics = {
            "roc_auc_score": roc_auc_score(Y, y_pred),
            "accuracy_score": accuracy_score(Y, (y_pred > 0.5).astype(int)),
        }

    used_df[f"{y}_pred"] = y_pred
    used_df[f"{y}_pred_zscore"] = (y_pred - y_pred.mean()) / y_pred.std()

    result_df = (
        model.summary2()
        .tables[1]
        .rename(
            columns={
                "Coef.": "coef",
                "Std.Err.": "std",
                "z": "z",
                "P>|z|": "pvalue",
                "[0.025": "upper_ci",
                "0.975]": "lower_ci",
            }
        )
    )
    result_df.index.name = "covariate"
    y_true = Y

    # result_df
    metrics = {
        "pearsonr": pearsonr(y_true, y_pred)[0],
        "spearmanr": spearmanr(y_true, y_pred)[0],
        "explained_variance_score": explained_variance_score(y_true, y_pred),
        "r2_score": r2_score(y_true, y_pred),
    }
    metrics.update(additional_metrics)

    return model, result_df, metrics, used_df