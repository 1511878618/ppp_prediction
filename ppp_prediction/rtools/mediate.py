# from . import tftoolsDir
from ppp_prediction.rtools import tftoolsDir
from typing import List, Optional, Union
import pandas as pd
from ppp_prediction.utils import parallel_df_decorator, dataframe_column_name_convert


def mediate_parallel(
    data: pd.DataFrame,
    X_Y_M_combination: pd.DataFrame,
    covariates: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    sims=500,
    ncores=None,
):
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri

    # define the R NULL object
    r_null = robjects.r("NULL")

    # 启用 R 和 pandas 数据转换功能
    pandas2ri.activate()

    # load local package
    r_local_package_dir = tftoolsDir
    # robjects.r("library(devtools)")
    robjects.r(
        """
    library(parallel)
    library(foreach)
    library(bruceR)
    library(doParallel)
    library(dplyr)           

    """
    )
    robjects.r(f'devtools::load_all("{r_local_package_dir}")')
    mediate_parallel = robjects.r("mediate_parallel")

    #
    res = mediate_parallel(
        data=data,
        X_Y_M_combination=X_Y_M_combination,
        covariates=robjects.StrVector(covariates) if covariates else r_null,
        cat_cols=robjects.StrVector(cat_cols) if cat_cols else r_null,
        sims=sims,
        ncores=ncores if ncores else r_null,
    )
    # format the data
    extracted_res = {k: pandas2ri.rpy2py(v) for k, v in res.items()}

    res_list = []
    for k, v in extracted_res.items():
        c_mediation_res_df = v.reset_index(drop=False, names=["MediationType"])
        name_list = k.split(".")
        if len(name_list) == 3:
            X, M, Y = name_list
            c_mediation_res_df["X"] = X
            c_mediation_res_df["M"] = M
            c_mediation_res_df["Y"] = Y
        c_mediation_res_df["name"] = k

        res_list.append(c_mediation_res_df)
    res_df = pd.concat(res_list)
    return res_df



@dataframe_column_name_convert(
    "data",
    to_check_col_params=["X", "Y", "M", "covariates", "cat_cols"],
    return_reconvert_col_params=[],
)
def mediate(
    data: pd.DataFrame,
    X,
    Y,
    M,
    covariates: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    boot=True,
    sims=500,
    norm_x: Optional[str] = None,
    M_logit: Optional[bool] = False,
    Y_logit: Optional[bool] = False,
):
    """
    Run the mediate function from the tftools package
    :param data: pandas DataFrame, the data to be analyzed
    :param X: str, the column name of the exposure variable
    :param Y: str, the column name of the outcome variable
    :param M: str, the column name of the mediator variable
    :param covariates: list, the list of covariates
    :param cat_cols: list, the list of categorical columns
    :param boot: bool, whether to use bootstrap
    :param sims: int, the number of simulations
    :return: pandas DataFrame
    """

    # import rpy2
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri

    # define the R NULL object
    r_null = robjects.r("NULL")

    # 启用 R 和 pandas 数据转换功能
    pandas2ri.activate()

    # load local package
    r_local_package_dir = tftoolsDir
    robjects.r("library(devtools)")
    robjects.r(f'devtools::load_all("{r_local_package_dir}")')

    # define the R function
    run_mediate = robjects.r["run_mediate"]

    need_cols = [X, Y, M]
    to_norm_col = [X, M]  # to normalize

    if covariates:
        need_cols += covariates
        to_norm_col = covariates
    if cat_cols:
        need_cols += cat_cols
        to_norm_col = [i for i in to_norm_col if i not in cat_cols]

    data = data[need_cols].copy().dropna()

    if norm_x:
        if norm_x == "zscore":
            data[to_norm_col] = (data[to_norm_col] - data[to_norm_col].mean()) / data[
                to_norm_col
            ].std()
        else:
            raise ValueError(
                f"norm_x must be 'zscore' or None, but got {norm_x} which is not supported"
            )

    r_data = pandas2ri.py2rpy(data)

    res_r_df = run_mediate(
        r_data,
        X=X,
        Y=Y,
        M=M,
        covariates=robjects.StrVector(covariates) if covariates else r_null,
        cat_cols=robjects.StrVector(cat_cols) if cat_cols else r_null,
        boot=boot,
        sims=sims,
        M_logit=M_logit,
        Y_logit=Y_logit,
    )

    res_df = pandas2ri.rpy2py(res_r_df)
    return res_df



