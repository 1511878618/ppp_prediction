import pandas as pd
import subprocess
import sys 
try:
    from rich.console import Console
    from rich.table import Table
except:
    print("缺少rich模块,开始安装...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "rich"], check=True)
        print("successfully installed rich")
        from rich.console import Console
        from rich.table import Table
    except subprocess.CalledProcessError:
        print("unable to install rich, please install it manually")
        sys.exit(1)

import os 

import re
import string

from functools import wraps
from typing import List, Optional


from ppp_prediction.cox import columnsFormatV1

from functools import wraps
from typing import List, Optional


def dataframe_column_name_convert(
    data_params,
    to_check_col_params: Optional[List[str]] = None,
    return_reconvert_col_params: Optional[List[str]] = None,
):
    """
    like run_cmprisk function accept run_cmprisk(data, exposure, outcome, time, covariates, cat_cols, outcome_order,...)
        1. data_params = "data",  # the data parameter name in the function
        2. to_check_col_params = ["exposure", "outcome", "time", "covariates", "cat_cols"], # the columns to check in the data parameter, which need to convert to No special characters
        3. the function will return a dataframe, and part of the columns need to convert back to the original column name

    Bugs:
    1. params should not be passed as a postional argument, should be passed as a keyword argument
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # print(args, kwargs)
            if data_params not in kwargs:
                raise ValueError(
                    f"No data found in the arguments to format with kwargs: {kwargs.keys()}"
                )
            # format the data columns
            dfFormat = columnsFormatV1(kwargs[data_params])
            formated_df = dfFormat.format(
                kwargs[data_params]
            ).copy()  # copy the data to avoid change the original data

            # check the columns in the data to convert
            for to_check_col in to_check_col_params:

                # make sure all to_check_col_params in the kwargs
                if to_check_col not in kwargs:
                    raise ValueError(
                        f"No {to_check_col} found in the arguments to format with kwargs: {kwargs.keys()}"
                    )

                # check if the to_check_col is a list

                to_check_col = kwargs[to_check_col]
                if isinstance(to_check_col, str):
                    to_check_col = [to_check_col]

                # make sure each params include the columns isin the data
                if to_check_col is not None:
                    for each_col in to_check_col:
                        if each_col not in kwargs[data_params].columns:
                            raise ValueError(
                                f"No {to_check_col} found in the data to format with columns: {kwargs[data_params].columns}"
                            )

            # updated at the same place with dfFormat
            for to_check_col in to_check_col_params:
                extracted_value = kwargs[to_check_col]
                if isinstance(extracted_value, str):
                    kwargs[to_check_col] = dfFormat.get_format_column(extracted_value)
                elif isinstance(extracted_value, list):
                    kwargs[to_check_col] = [
                        dfFormat.get_format_column(each_value)
                        for each_value in extracted_value
                    ]
                elif extracted_value is None:
                    pass
                else:
                    raise ValueError(
                        f"Unsupported type of {to_check_col} found in the arguments to format with kwargs: {kwargs.keys()}"
                    )

            # update the data with the formated data
            kwargs[data_params] = formated_df

            result = func(*args, **kwargs)

            for reconvert_col in return_reconvert_col_params:
                if reconvert_col not in result.columns:
                    raise ValueError(
                        f"No {reconvert_col} found in the result to reconvert, developer need to check the return_reconvert_col_params with the result columns: {result.columns} and the return_reconvert_col_params: {return_reconvert_col_params}"
                    )

                result[reconvert_col] = result[reconvert_col].apply(
                    lambda x: dfFormat.get_reverse_column(x),
                )

            return result

        return wrapper

    return decorator

def parallel_df_decorator(
    data_params, to_split_param, other_keep_param: Optional[List[str]] = None
):
    """
    A decorator to split the data by a column or a list of columns and run the function in parallel
    :param data_params: str, the parameter name of the data in the function, like `data`
    :param to_split_param: str, the parameter name of the column or a list of columns to split the data, like `exposure`
    :param other_keep_param: list, the list of other parameters to keep in the data, like `["outcome", "time"]`, which are the columns to keep in the data when split the data

    the function must have threads parameter to specify the number of threads to run the function in parallel

    Example:
    # define a single function
    def test_func(data, exposure, outcome, time, other_param):
        return data

    # use the decorator to parallel the function
    @parallel_df_decorator(data_params="data", to_split_param="exposure", other_keep_param=["outcome", "time"])
    def test_func_parallel(data, exposure, outcome, time, other_param, threads=4):
        return data

    # run the function
    test_func_parallel(data, exposure=["A", "B"], outcome="disease", time="survTime", other_param="other")

    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            # split the data
            # to_split_param is the parameter to keep columns name of the data to split, e.g. exposure

            if to_split_param in kwargs:
                split_by = kwargs[to_split_param]

                if isinstance(split_by, str):
                    raise ValueError(
                        f"{to_split_param} should be a list of columns to split the data"
                    )
            else:
                raise ValueError(f"{to_split_param} is not found in the arguments")

            # other keep params passed params to keep in the data each params should be a string or a list of strings that are the column names; e.g. ["outcome", "time"]
            to_keep_cols = []
            # if other_keep_param:

            if other_keep_param is not None:

                keep_params = {k: kwargs[k] for k in other_keep_param}
                to_keep_cols = []
                for k, v in keep_params.items():
                    if isinstance(v, str):
                        to_keep_cols.append(v)
                    elif isinstance(v, list):
                        to_keep_cols += v
                    else:
                        # raise ValueError(f"{k} should be a string or a list of strings")
                        pass

            #  extract data
            if data_params not in kwargs:
                raise ValueError(f"{data_params} is not found in the arguments")
            data = kwargs.pop(data_params)

            # pop the split_by param
            kwargs.pop(to_split_param)

            # check all cols are in the data
            for col in to_keep_cols:
                if col not in data.columns:
                    raise ValueError(f"{col} is not found in the data columns")
            for col in split_by:
                if col not in data.columns:
                    raise ValueError(f"{col} is not found in the data columns")

            # print(data_params, split_by, to_keep_cols)
            # for i in split_by:
            #     print(data[[i, *to_keep_cols]].head())
            threads = kwargs.get("threads", 1)
            if threads > 1:
                # parallel
                import multiprocessing
                from joblib import Parallel, delayed

                num_cores = multiprocessing.cpu_count()
                threads = min(threads, num_cores)
                print(f"Using {threads} threads to parallel")

                # run the function in parallel
                res = Parallel(n_jobs=threads)(
                    delayed(func)(
                        *args,
                        **{
                            data_params: data[
                                [i, *to_keep_cols]
                            ].copy(),  # only pass the data that is needed
                            to_split_param: i,
                        },
                        **kwargs,
                    )
                    for i in split_by
                )
                # return a list of dataframes to concat

            else:
                # not parallel

                res = [
                    func(
                        *args,
                        **{
                            data_params: data[
                                [i, *to_keep_cols]
                            ].copy(),  # only pass the data that is needed
                            to_split_param: i,
                        },
                        **kwargs,
                    )
                    for i in split_by
                ]

            res_df = pd.concat(res)

            return res_df

        return wrapper

    return decorator


# 替换字符串中的特殊字符
def replace_special_chars(text, special_chars="-+"):
    # 使用 translate 删除 string.punctuation 中的ASCII标点符号
    text = text.translate(
        str.maketrans("", "", "≥≤·！@#￥%……&*（）—+，。？、；：“”‘’《》{}【】")
    )

    # 使用正则表达式将特殊字符替换为下划线 "_"
    text = re.sub(f"[{re.escape(special_chars)}]", "_", text)

    # 替换空格为下划线
    text = text.replace(" ", "_")

    return text


def downsample_by(from_df, ref_df, by_cols, ratio=1):
    """
    Downsamples a DataFrame based on the frequency of values in another DataFrame.

    Parameters:
    from_df (DataFrame): The DataFrame to be downsampled.
    ref_df (DataFrame): The DataFrame used as a reference for downsampling.
    by_cols (str or list): The column(s) used for grouping and downsampling.

    Returns:
    DataFrame: The downsampled DataFrame.

    """
    if isinstance(by_cols, str):
        by_cols = [by_cols]

    from_df = from_df.copy()
    ref_df = ref_df.copy()
    dist = ref_df.groupby(by_cols).size()
    sampled_df = (
        from_df.groupby(by_cols, as_index=False)
        .apply(
            lambda x: x.sample(
                min(dist.get(x.name, 0) * ratio, x.shape[0]), random_state=1
            )
        )
        .reset_index(drop=True)
    )
    return sampled_df


def crosstab_multi(df, row_vars, col_vars):
    res = []
    for row_var in row_vars:
        col_res = []
        for col_var in col_vars:
            # 使用pd.crosstab计算每一对变量的交叉表，dropna=False保证包括NaN
            ct = pd.crosstab(
                df[row_var].fillna("Missing"),
                df[col_var].fillna("Missing"),
                dropna=False,
            )

            # 填充缺失值为0
            ct.fillna(0, inplace=True)

            # 重命名索引和列以提高可读性
            ct.index = [f"{row_var}_{idx}" for idx in ct.index]
            ct.columns = [f"{col_var}_{col}" for col in ct.columns]

            col_res.append(ct)

        # 合并当前行变量的所有列变量的结果
        if col_res:
            combined_col_res = pd.concat(col_res, axis=1)
            res.append(combined_col_res)

    # 合并所有行变量的结果
    if res:
        final_res = pd.concat(res)
        return final_res
    else:
        return pd.DataFrame()  # 返回一个空的DataFrame如果没有结果


def generate_qcut_labels(k):
    """
    k = 4 means <25%, 25%-50%, 50%-75%, >=75%
    k = 3 means <33%, 33%-66%, >=66%
    """
    res = []
    for i in range(k):
        if i == 0:

            res.append(f"<{(i+1)*100/k:.0f}%")
        elif i == k - 1:
            res.append(f">={(i)*100/k:.0f}%")
        else:
            res.append(f"{i*100/k:.0f}-{(i+1)*100/k:.0f}%")
    return res


def load_data(x, **kwargs):
    # if isinstance(x, Path.)
    x = str(x)
    if ".csv" in x:
        return pd.read_csv(x, **kwargs)
    elif x.endswith(".feather"):
        return pd.read_feather(x, **kwargs)
    elif x.endswith(".pkl"):
        return pd.read_pickle(x, **kwargs)
    elif ".tsv" in x:
        return pd.read_csv(x, sep="\t", **kwargs)
    else:
        raise ValueError(f"File format: {x} not supported")


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class DataFramePretty(object):
    def __init__(self, df: pd.DataFrame) -> None:
        self.data = df

    def get_pretty(self):
        table = Table()

        # self.data是原始数据
        # df 是用来显示的数据
        df = self.data.copy()
        for col in df.columns:
            df[col] = df[col].astype("str")
            table.add_column(col)

        for idx in range(len(df)):
            table.add_row(*df.iloc[idx].tolist())
        return table

    def show(self):
        table = self.get_pretty()
        console = Console()
        console.print(table)


def modelParametersNum(model):
    totalNum = sum([i.numel() for i in model.parameters()])
    print(f"模型总参数个数：{totalNum}\t占用的总显存为{totalNum*4/1024/1024:.2f}MB")
    return totalNum


def try_gpu():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    return device
