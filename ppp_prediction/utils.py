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
