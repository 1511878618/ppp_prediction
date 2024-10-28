import warnings
from itertools import product
from typing import List, Tuple, Union, overload

import confidenceinterval as ci
import numpy as np
import pandas as pd
import scipy.stats as ss
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.multitest as smm
from confidenceinterval.bootstrap import bootstrap_ci
from pandas import DataFrame
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    auc,
    explained_variance_score,
    f1_score,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from statsmodels.stats.multitest import multipletests

# from tqdm.rich import tqdm
from tqdm.notebook import tqdm


from ppp_prediction.cox import get_cat_var_name, columnsFormat


import re


def ReverseColumnName(name, dfFormat):
    if name.startswith("C("):
        # [T.x] return x
        Number = re.findall(r"\[(.*?)\]", name)[0].split(".")[1]

        raw_name = dfFormat.get_reverse_column(get_cat_var_name(name)) + "_" + Number

    else:
        raw_name = dfFormat.get_reverse_column(name)
    return raw_name


def rank_INT(series, c=3.0 / 8, stochastic=True):
    """Perform rank-based inverse normal transformation on pandas series.
    If stochastic is True ties are given rank randomly, otherwise ties will
    share the same value. NaN values are ignored.

    Args:
        param1 (pandas.Series):   Series of values to transform
        param2 (Optional[float]): Constand parameter (Bloms constant)
        param3 (Optional[bool]):  Whether to randomise rank of ties

    Returns:
        pandas.Series
    """

    # Check input
    assert isinstance(series, pd.Series)
    assert isinstance(c, float)
    assert isinstance(stochastic, bool)

    # Set seed
    np.random.seed(123)

    # Take original series indexes

    raw_series = series.copy()
    # Drop NaNs
    series = series.loc[~pd.isnull(series)]
    orig_idx = series.index

    # Get ranks
    if stochastic == True:
        # Shuffle by index
        series = series.loc[np.random.permutation(series.index)]
        # Get rank, ties are determined by their position in the series (hence
        # why we randomised the series)
        rank = ss.rankdata(series, method="ordinal")
    else:
        # Get rank, ties are averaged
        rank = ss.rankdata(series, method="average")

    # Convert numpy array back to series
    rank = pd.Series(rank, index=series.index)

    # Convert rank to normal distribution
    transformed = rank.apply(rank_to_normal, c=c, n=len(rank))

    # return transformed[orig_idx]
    raw_series[orig_idx] = transformed[orig_idx]
    return raw_series


def rank_to_normal(rank, c, n):
    # Standard quantile function
    x = (rank - c) / (n - 2 * c + 1)
    return ss.norm.ppf(x)


def cal_residual(df, x: List, y: str, plus_mean=True, return_model=False):
    """
    res_df = cal_residual(a, x=['age','sex'], y='ldl_a')
    res_df
    """

    print(f"passed data have {df.shape[0]} rows")
    used_df = df.copy().dropna(subset=[y] + x)
    X = sm.add_constant(used_df[x])
    Y = used_df[y]
    print(f"used data have {used_df.shape[0]} rows after dropna")

    model = sm.OLS(Y, X).fit()

    resid = model.resid
    if plus_mean:
        resid = resid + Y.mean()
    resid.name = f"{y}_residual"

    final = df.merge(resid, left_index=True, right_index=True, how="left")

    if return_model:
        return final, model
    else:
        return final


def generate_multipletests_result(df, pvalue_col="pvalue", alpha=0.05, method="fdr_bh"):
    """
        method : str
        Method used for testing and adjustment of pvalues. Can be either the full name or initial letters. Available methods are:

    bonferroni : one-step correction
    sidak : one-step correction
    holm-sidak : step down method using Sidak adjustments
    holm : step-down method using Bonferroni adjustments
    simes-hochberg : step-up method (independent)
    hommel : closed method based on Simes tests (non-negative)
    fdr_bh : Benjamini/Hochberg (non-negative)
    fdr_by : Benjamini/Yekutieli (negative)
    fdr_tsbh : two stage fdr correction (non-negative)
    fdr_tsbky : two stage fdr correction (non-negative)
    """
    df = df.copy()
    pvalue_series = df[pvalue_col]
    reject, pvals_corrected, _, _ = multipletests(
        pvalue_series, alpha=alpha, method=method
    )
    df["p_adj"] = pvals_corrected
    df["reject"] = reject
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
        "N": len(y),
        "N_case": y.sum(),
        "N_control": len(y) - y.sum(),
    }


def cal_qt_metrics(y_true, y_pred):
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


# def cal_corr_multivar_v2(
#     df: DataFrame,
#     x: Union[str, List[str]],
#     y: str,
#     cofounder: Union[str, List[str]] = None,
#     adjust: bool = False,
#     norm_x=None,
#     model_type: Union[str, List[str]] = "glm",
#     family=sm.families.Gaussian(),
#     ci=False,
#     n_resamples=100,
#     verbose=False,
# ):
#     """
#     Calculate the correlation between x and y variables in the input DataFrame.

#     if cofounder will add to x, if adjust is True, will adjust for cofounder: res of (y~cofounder) and res ~ x

#     """

#     if cofounder is not None:
#         cofounder = [cofounder] if isinstance(cofounder, str) else cofounder
#     else:
#         cofounder = []

#     if isinstance(x, str):
#         x = [x]
#     used_df = df[x + [y] + cofounder].copy().dropna(how="any")
#     # Note the binary cofounder may be a single value as dropna or data is a subset, so drop them
#     for col in cofounder:
#         if used_df[col].nunique() <= 1:
#             used_df.drop(col, axis=1, inplace=True)
#             cofounder.remove(col)
#             print(f"drop {col} as binary variable without variance")

#     if norm_x == "zscore":
#         print(f"normalizing x={x} by zscore")
#         used_df[x] = (used_df[x] - used_df[x].mean()) / used_df[x].std()
#     elif norm_x == "int":
#         print(f"normalizing x={x} by rank inverse normal transformation")
#         used_df[x] = rank_INT(used_df[x])
#     else:
#         pass

#     X = sm.add_constant(used_df[x + cofounder])
#     Y = used_df[y]

#     if adjust:
#         Y = cal_residual(used_df, x=cofounder, y=y)[f"{y}_residual"]
#         X = sm.add_constant(used_df[[x]])

#     if model_type == "logistic" and adjust:
#         raise ValueError(
#             "adjust not support for logistic model, so use ols or glm instead"
#         )

#     if model_type == "glm":
#         model = sm.GLM(Y, X, family=family).fit()
#         y_pred = model.predict(X)
#         metrics = cal_qt_metrics(Y, y_pred)

#         # metrics = {"Persudo_R2": model.pseudo_rsquared()}
#         # metrics.update(cal_qt_metrics(Y, y_pred))
#     elif model_type == "ols":
#         model = sm.OLS(Y, X).fit()
#         y_pred = model.predict(X)

#         # metrics = {"Persudo_R2": model.pseudo_rsquared()}
#         # metrics.update(cal_qt_metrics(Y, y_pred))
#         metrics = cal_qt_metrics(Y, y_pred)
#     elif model_type == "logistic":
#         model = sm.Logit(Y, X).fit()
#         y_pred = model.predict(X)
#         if ci:
#             metrics = cal_binary_metrics_bootstrap(
#                 Y, y_pred, ci_kwargs={"n_resamples": n_resamples}
#             )
#         else:
#             metrics = cal_binary_metrics(Y, y_pred)
#     else:
#         raise ValueError(f"model_type {model_type} not supported")

#     model_res = (
#         model.summary2()
#         .tables[1]
#         .rename(
#             columns={
#                 "Coef.": "coef",
#                 "Std.Err.": "std",
#                 "z": "z",
#                 "P>|z|": "pvalue",
#                 "[0.025": "lower_ci",
#                 "0.975]": "upper_ci",
#             }
#         )
#     )
#     model_res.reset_index(drop=False, inplace=True, names="var")
#     if model_type == "logistic":
#         model_res["OR"] = np.exp(model_res["coef"])
#         model_res["OR_UCI"] = np.exp(model_res["upper_ci"])
#         model_res["OR_LCI"] = np.exp(model_res["lower_ci"])

#     if len(used_df[y].unique()) <= 2:
#         model_res["n_case"] = used_df[y].sum()
#         model_res["n_control"] = used_df.shape[0] - used_df[y].sum()
#     else:
#         model_res["N"] = used_df.shape[0]

#     for k, v in metrics.items():
#         model_res[k] = v

#     return model_res.iloc[1:, :], metrics  # drop intercept


def cal_corr_v3(
    df: DataFrame,
    x: Union[str, List[str]],
    y: Union[str, List[str]],
    cov: Union[str, List[str], None] = None,
    cat_cols: Union[List[str], None] = None,
    adjust: bool = False,
    norm_x=None,
    model_type: Union[str, List[str]] = "auto",
    threads: int = 4,
    verbose=False,
    return_all=False,
):
    # """
    # Calculate the correlation between variables x and y in the given DataFrame.

    # Parameters:
    # - df (DataFrame): The input DataFrame.
    # - x (Union[str, List[str]]): The x variable(s) for the correlation calculation.
    # - y (Union[str, List[str]]): The y variable(s) for the correlation calculation.
    # - cofounder (Union[str, List[str]], optional): The cofounder variable(s) to adjust for in the correlation calculation. Default is None.
    # - adjust (bool, optional): Whether to adjust the correlation for cofounders. Default is True.
    # - model_type (Union[str, List[str]], optional): The type of model to use for the correlation calculation. Default is "glm", ols or logistic is ok .
    # - threads (int, optional): The number of threads to use for parallel computation. Default is 4.
    # - family (object, optional): The family object specifying the distribution of the dependent variable. Default is sm.families.Gaussian().
    # - verbose (bool, optional): Whether to print verbose output. Default is False.
    # - norm_x (str, optional): The method to normalize x, default is None, which means no normalization. Other options are "zscore" and "int" for rank inverse normal transformation.

    # Returns:
    # - DataFrame: The correlation results.

    # Raises:
    # - ValueError: If x and y are not both str or list of str.

    # """

    if isinstance(x, list) or isinstance(y, list):
        df = df.copy()  # avoid inplace change
        model_type = [model_type] if isinstance(model_type, str) else model_type

        if isinstance(x, str):
            x = [x]
        if isinstance(y, str):
            y = [y]

        if isinstance(cov, str):
            cov = [cov]
        elif cov is None:
            cov = []

        if cat_cols is None:
            cat_cols = []
        elif isinstance(cat_cols, str):
            cat_cols = [cat_cols]

        x_y_model_combination = list(product(x, y, model_type))

        print(
            f"total {len(x_y_model_combination)} combination of  to cal by threads {threads}"
        )

        if threads == 1:
            res = [
                cal_corr_v3(
                    df[[x, y] + cov] if isinstance(x, str) else df[[*x, y] + cov],
                    x,
                    y,
                    cov,
                    cat_cols,
                    adjust,
                    norm_x,
                    current_model_type,
                    threads,
                    verbose,
                    return_all,
                )
                for x, y, current_model_type in tqdm(
                    x_y_model_combination,
                    total=len(x_y_model_combination),
                    desc="cal corrs",
                )
            ]

        else:
            from joblib import Parallel, delayed

            # res = Parallel(n_jobs=threads)(delayed(cal_corr_v2)(df[[x, y] + cofounder], x, y, cofounder, adjust,norm_x,current_model_type, 1, family, verbose) for x,y,current_model_type in tqdm(x_y_model_combination, total=len(x_y_model_combination), desc="cal corrs"))
            res = Parallel(n_jobs=threads)(
                delayed(cal_corr_v3)(
                    df[[x, y] + cov] if isinstance(x, str) else df[[*x, y] + cov],
                    x,
                    y,
                    cov,
                    cat_cols,
                    adjust,
                    norm_x,
                    current_model_type,
                    1,
                    verbose,
                    return_all,
                )
                for x, y, current_model_type in x_y_model_combination
            )

        return pd.concat(res)

    elif isinstance(x, str) and isinstance(y, str) and isinstance(model_type, str):
        try:
            if isinstance(cov, str):
                cov = [cov]
            elif cov is None:
                cov = []

            if cat_cols is None:
                cat_cols = []
            elif isinstance(cat_cols, str):
                cat_cols = [cat_cols]

            if threads > 1:
                print(
                    f"threads is {threads} and x is {x} and y is {y}, which is str, so don't supported for multi threads"
                )
            record = {}

            raw_df = df[[x, y] + cov].copy().dropna(how="any")

            # if norm_x is not None and norm data
            if norm_x is not None:
                to_norm = [i for i in [x] + cov if i not in cat_cols]
                if len(to_norm) >0:
                    if norm_x == "zscore":
                        print(f"normalizing x={x} by zscore")
                        raw_df[to_norm] = (
                            raw_df[to_norm] - raw_df[to_norm].mean()
                        ) / raw_df[to_norm].std()

                    elif norm_x == "int":
                        print(f"normalizing x={x} by rank inverse normal transformation")
                        raw_df[to_norm] = rank_INT(raw_df[to_norm])

                    else:
                        pass

                    record["norm_x"] = norm_x
                else:
                    record["norm_x"] = "no_qt_var"
                    print("no quantitative var to norm")

            # adjust for data
            if adjust:
                if len(cov) <= 0:
                    raise ValueError("adjust=True but no cov found")

                raw_df[y] = cal_residual(raw_df, x=cov, y=y)[f"{y}_residual"]
                record["adjust"] = 1

            # auto model type
            if model_type == "auto":
                if not adjust:
                    if len(raw_df[y].unique()) <= 2:
                        model_type = "logistic"
                    else:
                        model_type = "glm"

                    print(
                        f"auto model selection for x={x} and y={y} with model_type={model_type}"
                    )
                else:
                    raise ValueError("auto model selection not support for adjust=True")
                record["model_type"] = model_type
            # record
            raw_var_dict = {
                "x": x,
                "y": y,
                "cov": cov,
                "cat_cols": cat_cols,
            }

            # formated df
            dfFormat = columnsFormat(raw_df)  # to avoid space or special in column name
            formatted_df = dfFormat.format(raw_df)
            formatted_var_dict = {
                "x": dfFormat.get_format_column(x),
                "y": dfFormat.get_format_column(y),
                "cov": dfFormat.get_format_column(cov),
                "cat_cols": dfFormat.get_format_column(cat_cols),
            }

            X = sm.add_constant(
                formatted_df[[formatted_var_dict["x"]] + formatted_var_dict["cov"]]
            )
            Y = formatted_df[formatted_var_dict["y"]]

            # if dfFormat.get_reverse_column(formatted_var_dict["y"]) in cat_cols:

            X_str = " + ".join(
                [
                    f"C({col})" if col in formatted_var_dict["cat_cols"] else col
                    for col in [formatted_var_dict["x"]]
                ]
            )

            Cov_str = " + ".join(
                [
                    f"C({col})" if col in formatted_var_dict["cat_cols"] else col
                    for col in formatted_var_dict["cov"]
                ]
            )

            Y_str = formatted_var_dict["y"]

            # generate formula
            formula = f"{Y_str} ~ {X_str} + {Cov_str}" if len(cov) > 0 else f"{Y_str} ~ {X_str}"
            print(formula)

            if model_type == "glm":

                # default_fit_params = {"disp": False}
                # for k in default_fit_params:
                #     if k not in fit_params:
                #         fit_params[k] = default_fit_params[k]

                # model = sm.GLM(Y, X, family=family).fit(**fit_params)
                model = smf.glm(
                    formula,
                    data=formatted_df,
                ).fit(disp=False)
                y_pred = model.predict(X)
                metrics = cal_qt_metrics(Y, y_pred)

            elif model_type == "logistic":

                # nacse
                ncase = formatted_df[formatted_var_dict["y"]].sum()
                if ncase <= 2:
                    warnings.warn(f"n_case is {ncase} for y={y}, which is less than 5")
                    return pd.DataFrame(
                        {
                            "var": [x],
                            "exposure": [y],
                            "cov": " + ".join(cov),
                            "model": [model_type],
                            "adjust_cov": [1 if adjust else 0],
                            "norm_x": [norm_x],
                            "n_case": [ncase],
                            "error": [f"n_caseError {ncase}"],
                        }
                    )
                    # return
                # fit_params.update({"maxiter": 100})
                # Reason: some params may not found solve, so try many
                fit_params_list = [
                    *[
                        dict(
                            method="l1",
                            alpha=alpha,
                        )
                        for alpha in [0, 0.05, 0.1, 0.2, 0.3]
                    ],
                    *[
                        dict(
                            method="l1",
                            alpha=alpha,
                            disp=0,
                            trim_mode="size",
                            qc_verbose=0,
                        )
                        for alpha in [0.1, 0.2]
                    ],
                ]
                status = 0
                for fit_params in fit_params_list:
                    try:
                        model = smf.logit(
                            formula,
                            data=formatted_df,
                        ).fit_regularized(**fit_params)
                        # pvalue = model.pvalues[X_str]
                        # if np.isnan(pvalue):
                        #     continue
                        status = 1
                        break
                    except Exception as e:
                        print(f"error for {fit_params} with {str(e)}")
                        continue
                if status == 0:
                    return pd.DataFrame(
                        {
                            "var": [x],
                            "exposure": [y],
                            "cov": " + ".join(cov),
                            "model": [model_type],
                            "adjust_cov": [1 if adjust else 0],
                            "norm_x": [norm_x],
                            "n_case": [ncase],
                            "error": ["fit error"],
                        }
                    )


                y_pred = model.predict(X)
                metrics = cal_binary_metrics(Y, y_pred)
                # metrics["fit_params"] = str(fit_params)
            else:
                raise ValueError(f"model_type {model_type} not supported")

            res_df = (
                model.summary2()
                .tables[1]
                .rename(
                    columns={
                        "Coef.": "coef",
                        "Std.Err.": "std",
                        "z": "z",
                        "P>|z|": "pvalue",
                        "[0.025": "lower_ci",
                        "0.975]": "upper_ci",
                    },
                )
            ).reset_index(drop=False, names="var")

            res_df["var"] = res_df["var"].apply(
                lambda x: ReverseColumnName(x, dfFormat)
            )

            if model_type == "logistic":
                res_df["OR"] = np.exp(res_df["coef"])
                res_df["OR_UCI"] = np.exp(res_df["upper_ci"])
                res_df["OR_LCI"] = np.exp(res_df["lower_ci"])

                res_df["n_case"] = res_df["var"].apply(
                    lambda x: (
                        raw_df[
                            raw_df[raw_var_dict["y"]]
                            == 1 & raw_df[raw_var_dict[x]]
                            == 1
                        ].shape[0]
                        if x in raw_var_dict["cat_cols"]
                        else None
                    )
                )
                res_df["n_control"] = res_df["var"].apply(
                    lambda x: (
                        raw_df[
                            raw_df[raw_var_dict["y"]]
                            == 0 & raw_df[raw_var_dict[x]]
                            == 1
                        ].shape[0]
                        if x in raw_var_dict["cat_cols"]
                        else None
                    )
                )

                res_df["N"] = raw_df.shape[0]

            else:
                res_df["N"] = raw_df.shape[0]

            for k, v in metrics.items():
                res_df[k] = v
            for k, v in record.items():
                res_df[k] = v

            res_df["x"] = x
            res_df["y"] = y
            res_df["cov"] = " + ".join(cov)

            res_df = res_df.set_index(
                ["x", "y", "cov"],
            ).reset_index()

            if not return_all:

                res_df = res_df[
                    res_df["var"].str.contains("|".join([x]), case=False, na=False)
                ]

            return res_df

        except Exception as e:
            print(
                f"Error for x={x} and y={y} and model_type={model_type} with {str(e)}"
            )
            return pd.DataFrame(
                {
                    "var": [x],
                    "exposure": [y],
                    "cov": " + ".join(cov),
                    "model": [model_type],
                    "adjust_cov": [1 if adjust else 0],
                    "norm_x": [norm_x],
                    "error": [str(e)],
                }
            )
            # raise e

    else:
        raise ValueError(
            "x and y  and model_type should be all str or list of str,"
            + f"but is {type(x)}, {type(y)}, {type(model_type)}"
            + "not support for other type"
        )


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
