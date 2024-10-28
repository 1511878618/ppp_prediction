# from . import tftoolsDir
from ppp_prediction.rtools import tftoolsDir
from typing import List, Optional, Union
import pandas as pd
from ppp_prediction.utils import parallel_df_decorator, dataframe_column_name_convert



@dataframe_column_name_convert(
    "data",
    to_check_col_params=["exposure", "outcome", "time", "covariates", "cat_cols"],
    return_reconvert_col_params=["var"],
)
def cmprisk(
    data: pd.DataFrame,
    exposure: str,
    outcome: str,
    time: str,
    outcome_order: List[str],
    covariates: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    norm_x: Optional[str] = None,
    saveDir: Optional[str] = None,
):
    """
    Run the cmprisk function from the tftools package
    :param data: pandas DataFrame, the data to be analyzed
    :param exposure: str, the column name of the exposure variable
    :param outcome: str, the column name of the outcome variable
    :param time: str, the column name of the time variable
    :param outcome_order: list, the order of the outcome variable
    :param covariates: list, the list of covariates
    :param cat_cols: list, the list of categorical columns

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
    run_cmprisk = robjects.r["run_cmprisk"]

    need_cols = [exposure, outcome, time]
    to_norm_col = [exposure]
    if covariates:
        need_cols += covariates
        to_norm_col += covariates
    if cat_cols:
        need_cols += cat_cols
        to_norm_col = [i for i in to_norm_col if i not in cat_cols]
    print(need_cols)
    data = data[need_cols].copy().dropna()
    if norm_x:
        if norm_x == "zscore":
            data[to_norm_col] = (data[to_norm_col] - data[to_norm_col].mean()) / data[to_norm_col].std()
        else:
            raise ValueError(f"norm_x must be 'zscore' or None, but got {norm_x} which is not supported")

    r_data = pandas2ri.py2rpy(data)

    ## run the function
    res_r_df = run_cmprisk(
        r_data,
        exposure=exposure,
        outcome=outcome,
        time=time,
        covariates=robjects.StrVector(covariates) if covariates else r_null,
        cat_cols=robjects.StrVector(cat_cols) if cat_cols else r_null,
        outcome_order=robjects.StrVector(outcome_order),
        saveDir=saveDir if saveDir else r_null,
    )

    res_df = pandas2ri.rpy2py(res_r_df)
    return res_df

@parallel_df_decorator(
    "data", "exposure", other_keep_param=["outcome", "time", "covariates"]
)
def cmprisk_parallel(
    data: pd.DataFrame,
    exposure: str,
    outcome: str,
    time: str,
    outcome_order: List[str],
    covariates: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    saveDir: Optional[str] = None,
    norm_x: Optional[str] = None,
    threads: Optional[int] = None,
):
    return cmprisk(
        data = data,
        exposure=exposure,
        outcome=outcome,
        time=time,
        outcome_order=outcome_order,
        covariates=covariates,
        cat_cols=cat_cols,
        saveDir=saveDir,
        norm_x=norm_x,
    )
