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
# from tqdm.rich import tqdm
from tqdm.notebook import tqdm
import numpy as np
from confidenceinterval.bootstrap import bootstrap_ci
import confidenceinterval as ci
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
from confidenceinterval.bootstrap import bootstrap_ci
import confidenceinterval as ci
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


def rank_INT(series, c=3.0/8, stochastic=True):
    """ Perform rank-based inverse normal transformation on pandas series.
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
    assert(isinstance(series, pd.Series))
    assert(isinstance(c, float))
    assert(isinstance(stochastic, bool))

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
    x = (rank - c) / (n - 2*c + 1)
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


def generate_multipletests_result(df, pvalue_col='pvalue',alpha=0.05, method='fdr_bh'):
    df = df.copy() 
    pvalue_series = df[pvalue_col]
    reject, pvals_corrected, _, _ = multipletests(pvalue_series, alpha=alpha, method=method)
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
    verbose=False,
) -> pd.DataFrame:
    if isinstance(x, list):
        if parallel_cores is not None and return_all == False:

            from joblib import Parallel, delayed

            x_split = split_list_into_k_chunks(x, k=parallel_cores)
            df_split = [df[x_ + cofounders + [y]] for x_ in x_split]

            res = Parallel(n_jobs=parallel_cores)(
                delayed(cal_corr)(
                    df_current,
                    x_current,
                    y,
                    cofounders,
                    model_type,
                    family,
                    return_all,
                    None,
                    0,
                    False,
                )
                for df_current, x_current in zip(df_split, x_split)
            )
            return pd.concat(res)

        result = []
        for x_ in tqdm(x, desc="processing", total=len(x)):
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
    print(f"passed data have {df.shape[0]} rows") if verbose else None
    used_df = df[[x, y] + cofounders].dropna()

    X = sm.add_constant(used_df[[x] + cofounders])
    Y = used_df[y]
    print(f"used data have {used_df.shape[0]} rows after dropna") if verbose else None
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


def cal_corr_multivar_v2(
    df: DataFrame,
    x: Union[str, List[str]],
    y: str,
    cofounder: Union[str, List[str]] = None,
    adjust: bool = False,
    norm_x=None,
    model_type: Union[str, List[str]] = "glm",
    family=sm.families.Gaussian(),
    ci= False,
    n_resamples = 100,
    verbose=False,
):
    """
    Calculate the correlation between x and y variables in the input DataFrame.

    if cofounder will add to x, if adjust is True, will adjust for cofounder: res of (y~cofounder) and res ~ x

    """

    if cofounder is not None:
        cofounder = [cofounder] if isinstance(cofounder, str) else cofounder
    else:
        cofounder = []

    if isinstance(x, str):
        x = [x]
    used_df = df[x + [y] + cofounder].copy().dropna(how="any")

    if norm_x == "zscore":
        print(f"normalizing x={x} by zscore")
        used_df[x] = (used_df[x] - used_df[x].mean()) / used_df[x].std()
    elif norm_x == "int":
        print(f"normalizing x={x} by rank inverse normal transformation")
        used_df[x] = rank_INT(used_df[x])
    else:
        pass

    X = sm.add_constant(used_df[x + cofounder])
    Y = used_df[y]

    if adjust:
        Y = cal_residual(used_df, x=cofounder, y=y)[f"{y}_residual"]
        X = sm.add_constant(used_df[[x]])

    if model_type == "logistic" and adjust:
        raise ValueError(
            "adjust not support for logistic model, so use ols or glm instead"
        )

    if model_type == "glm":
        model = sm.GLM(Y, X, family=family).fit()
        y_pred = model.predict(X)
        metrics = cal_qt_metrics(Y, y_pred)

        # metrics = {"Persudo_R2": model.pseudo_rsquared()}
        # metrics.update(cal_qt_metrics(Y, y_pred))
    elif model_type == "ols":
        model = sm.OLS(Y, X).fit()
        y_pred = model.predict(X)

        # metrics = {"Persudo_R2": model.pseudo_rsquared()}
        # metrics.update(cal_qt_metrics(Y, y_pred))
        metrics = cal_qt_metrics(Y, y_pred)
    elif model_type == "logistic":
        model = sm.Logit(Y, X).fit()
        y_pred = model.predict(X)
        if ci:
            metrics = cal_binary_metrics_bootstrap(Y, y_pred, ci_kwargs={"n_resamples": n_resamples})
        else:   
            metrics = cal_binary_metrics(Y, y_pred)
    else:
        raise ValueError(f"model_type {model_type} not supported")

    model_res = (
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
            }
        )
    )
    model_res.reset_index(drop=False, inplace=True, names="var")
    if model_type == "logistic":
        model_res["OR"] = np.exp(model_res["coef"])
        model_res["OR_UCI"] = np.exp(model_res["upper_ci"])
        model_res["OR_LCI"] = np.exp(model_res["lower_ci"])

    if len(used_df[y].unique()) <= 2:
        model_res["n_case"] = used_df[y].sum()
        model_res["n_control"] = used_df.shape[0] - used_df[y].sum()
    else:
        model_res["N"] = used_df.shape[0]
    
    for k, v in metrics.items():
        model_res[k] = v

    return model_res.iloc[1:, :], metrics  # drop intercept

def cal_corr_v2(
        df:DataFrame,
        x:Union[str, List[str]],
        y:Union[str, List[str]],
        cofounder:Union[str, List[str]]=None,
        adjust:bool=False,
        norm_x=None,
        model_type:Union[str, List[str]]="glm",
        threads:int=4,
        family=sm.families.Gaussian(),
        verbose = False,
):
    """
    Calculate the correlation between variables x and y in the given DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - x (Union[str, List[str]]): The x variable(s) for the correlation calculation.
    - y (Union[str, List[str]]): The y variable(s) for the correlation calculation.
    - cofounder (Union[str, List[str]], optional): The cofounder variable(s) to adjust for in the correlation calculation. Default is None.
    - adjust (bool, optional): Whether to adjust the correlation for cofounders. Default is True.
    - model_type (Union[str, List[str]], optional): The type of model to use for the correlation calculation. Default is "glm", ols or logistic is ok .
    - threads (int, optional): The number of threads to use for parallel computation. Default is 4.
    - family (object, optional): The family object specifying the distribution of the dependent variable. Default is sm.families.Gaussian().
    - verbose (bool, optional): Whether to print verbose output. Default is False.
    - norm_x (str, optional): The method to normalize x, default is None, which means no normalization. Other options are "zscore" and "int" for rank inverse normal transformation.

    Returns:
    - DataFrame: The correlation results.

    Raises:
    - ValueError: If x and y are not both str or list of str.

    """

    if isinstance(x, list) or isinstance(y, list):
        df = df.copy()  # avoid inplace change
        model_type = [model_type] if isinstance(model_type, str) else model_type

        if isinstance(x, str):
            x = [x]
        if isinstance(y, str):
            y = [y]

        x_y_model_combination = list(product(x, y, model_type))
        print(f"total {len(x_y_model_combination)} combination of  to cal by threads {threads}")
        if threads ==1:
            res = [cal_corr_v2(df[[x, y] + cofounder], x, y, cofounder,adjust,norm_x, current_model_type, threads, family,verbose) for x,y,current_model_type in tqdm(x_y_model_combination, total=len(x_y_model_combination), desc="cal corrs")]

        else:
            from joblib import Parallel, delayed
            res = Parallel(n_jobs=threads)(delayed(cal_corr_v2)(df[[x, y] + cofounder], x, y, cofounder, adjust,norm_x,current_model_type, 1, family, verbose) for x,y,current_model_type in tqdm(x_y_model_combination, total=len(x_y_model_combination), desc="cal corrs"))

        return pd.concat(res, axis=1).T

    elif isinstance(x, str) and isinstance(y, str) and isinstance(model_type, str):
        try:
            if cofounder is not None:
                cofounder = [cofounder] if isinstance(cofounder, str) else cofounder
            else:
                cofounder = []
            if threads >1:
                print(f"threads is {threads} and x is {x} and y is {y}, which is str, so don't supported for multi threads")

            used_df = df[[x, y] + cofounder].copy().dropna(how="any")

            if norm_x == "zscore":
                print(f"normalizing x={x} by zscore")
                used_df[x] = (used_df[x] - used_df[x].mean()) / used_df[x].std()
            elif norm_x == "int":
                print(f"normalizing x={x} by rank inverse normal transformation")
                used_df[x] = rank_INT(used_df[x])
            else:
                pass

            X = sm.add_constant(used_df[[x] + cofounder])
            Y = used_df[y]

            if adjust:
                Y = cal_residual(used_df, x=cofounder, y=y)[f"{y}_residual"]
                X = sm.add_constant(used_df[[x]])

            if model_type == "auto":
                if not adjust:
                    if len(used_df[y].unique()) <= 2:
                        model_type = "logistic"
                    else:
                        model_type = "glm"

                    print(f"auto model selection for x={x} and y={y} with model_type={model_type}")
                else:
                    raise ValueError("auto model selection not support for adjust=True")

            if model_type == "logistic" and adjust:
                raise ValueError(
                    "adjust not support for logistic model, so use ols or glm instead"
                )

            if model_type == "glm":
                model = sm.GLM(Y, X, family=family).fit()
                y_pred = model.predict(X)
                metrics = cal_qt_metrics(Y, y_pred)

                # metrics = {"Persudo_R2": model.pseudo_rsquared()}
                # metrics.update(cal_qt_metrics(Y, y_pred))
            elif model_type == "ols":
                model = sm.OLS(Y, X).fit()
                y_pred = model.predict(X)

                # metrics = {"Persudo_R2": model.pseudo_rsquared()}
                # metrics.update(cal_qt_metrics(Y, y_pred))
                metrics = cal_qt_metrics(Y, y_pred)
            elif model_type == "logistic":
                model = sm.Logit(Y, X).fit()
                y_pred = model.predict(X)
                metrics = cal_binary_metrics(Y, y_pred)
            else:
                raise ValueError(f"model_type {model_type} not supported")

            model_res = (
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
                    }
                )
                .loc[x]
                .to_dict()
            )
            result = {
                "var": x,
                "exposure": y,
                "model": model_type,
                "adjust_cov":1 if adjust else 0,
                "norm_x": norm_x,
                "pvalue": model.pvalues[x],
                "coef": model.params[x],
                "std": model_res["std"],
                "z": model_res["z"] if "z" in model_res else None,
                "upper": model_res["upper_ci"],
                "lower": model_res["lower_ci"],
            }

            if model_type == "logistic":
                result['OR'] = np.exp(result['coef'])
                result['OR_UCI'] = np.exp(result['upper'])
                result['OR_LCI'] = np.exp(result['lower'])

            if len(cofounder) <=0:
                result.update(metrics)

            if len(used_df[y].unique()) <= 2:
                case_control_metrics = {
                    "n_case": used_df[y].sum(),
                    "n_control": used_df.shape[0] - used_df[y].sum(),
                }
                result.update(case_control_metrics)
                result.update({"N": used_df.shape[0]})
            else:
                result.update({"N": used_df.shape[0]})

            res_series = pd.Series(result)
            return res_series

        except Exception as e:
            print(f"Error for x={x} and y={y} and model_type={model_type} with {str(e)[:100]}")

            raise e

            return pd.Series({
                "var": x,
                "exposure": y,
                "model": model_type,
                "adjust_cov":1 if adjust else 0,
                "norm_x": norm_x,
                
            })

    else:
        raise ValueError("x and y  and model_type should be all str or list of str," + f"but is {type(x)}, {type(y)}, {type(model_type)}"+"not support for other type")


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
    if return_all:
        return model, result_df, metrics, used_df
    else:
        return result_df

import matplotlib.pyplot as plt
from skmisc.loess import loess
import seaborn as sns

def plot_corrs(
    data,
    x,
    y,
    cofounders=None,
    beta=None,
    pvalue=None,
    scatter_kw=None,
    line_kw = None,
    title=None,
    ax=None,
    model_type=None,
):
    # 如果没有传入 ax，就创建一个新的
    if line_kw is None:
        line_kw = {}
    if ax is None:
        fig, ax = plt.subplots()
    title = ""
    # 定义默认参数
    default_scatter_kw = {"alpha": 0.5, "edgecolors": "none"}
    data = data[[x, y]].copy()
    # 更新参数（如果用户提供了自定义参数）
    if scatter_kw is not None:
        default_scatter_kw.update(scatter_kw)

    title += f"{x} vs {y}"
    if beta is None and pvalue is None:

        model_type = "glm" if model_type is None else model_type
        title += f" {model_type} model"

        if cofounders:
            title += f" with {cofounders[:3]} etc."

        model, res_df, used_df = cal_corr(
            data, x, y, cofounders=cofounders, model_type=model_type, return_all=True
        )
        # loess_model = loess(used_df[y].values, used_df[f"{y}_pred"])

        # data["loess"] = loess_model.predict(data[y].values).values

        beta = res_df["coef"]
        r = res_df["pearsonr"]
        pvalue = res_df["pvalue"]
        legend_text = f"Beta={beta:.2f}\nR={r:.2f}\np={pvalue:.2e}"

    else:
        legend_text = f"Beta={beta:.2f}\np={pvalue:.2e}"
        title += f"{x} vs {y}"
    # 绘制散点图
    # line_kw.pop("ci")
    sns.lineplot(data=data, x=x, y=y, ax=ax, color="black", ci = None, **line_kw)

    ax.scatter(data[x], data[y], **default_scatter_kw)

    # 添加相关性系数和 p 值
    # 构建显示在图例中的文本
    # 创建一个假的点（不可见的），仅用于在图例中显示相关性系数和 p 值
    ax.plot([], [], " ", label=legend_text)

    # 添加图例，利用 loc='best' 让 matplotlib 自动选择最佳位置
    # ax.legend(loc='best')
    ax.legend(
        loc="best",
        handlelength=0,
        handletextpad=0,
        frameon=False,
        fontsize="medium",
        labelspacing=1.2,
    )

    # 添加标题和轴标签

    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    # 返回ax对象
    return ax
