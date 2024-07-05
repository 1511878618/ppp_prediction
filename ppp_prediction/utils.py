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

    def show(self):
        table = Table()

        # self.data是原始数据
        # df 是用来显示的数据
        df = self.data.copy()
        for col in df.columns:
            df[col] = df[col].astype("str")
            table.add_column(col)

        for idx in range(len(df)):
            table.add_row(*df.iloc[idx].tolist())

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
