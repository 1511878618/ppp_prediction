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
