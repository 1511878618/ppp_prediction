#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       :
@Date     :2024/05/11 16:56:58
@Author      :Tingfeng Xu
@version      :1.0
'''
import argparse
from scipy.stats import pearsonr
import pandas as pd 
from pathlib import Path 
from multiprocessing import Pool
import math 
from functools import partial
import os 

import time 
import warnings
import textwrap

import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

import numpy as np
import pandas as pd
import scipy.stats as ss
from ppp_prediction.corr import cal_corr_v2
from ppp_prediction.utils import load_data

warnings.filterwarnings("ignore")
import time 


def generate_multipletests_result(df, pvalue_col='pvalue',alpha=0.05, method='fdr_bh'):
    df = df.copy() 
    pvalue_series = df[pvalue_col]
    reject, pvals_corrected, _, _ = multipletests(pvalue_series, alpha=alpha, method='fdr_bh')
    df['pval_corrected'] = pvals_corrected
    df['reject'] = reject
    return df 


class Timing(object):
    """
    计时器
    """
    def __init__(self):
        self.start = time.time()

    def __call__(self):
        return time.time() - self.start 

    def __str__(self):
        return str(self.__call__())

    def __repr__(self):
        return str(self.__call__())

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


def read_data(path:str):
    """
    for pickle or csv files only 
    """
    if path.endswith(".pkl"):
        return pd.read_pickle(path)
    elif path.endswith(".tsv"):
        return pd.read_csv(path, sep="\s+")
    else:
        return pd.read_csv(path)

def get_name(x):
    return Path(x).name


def average_list(x, nums = 5):
    """
    a = range(10)
    split_list = average_list(a, 2)
    split_list => [[0, 1,2,3,4], [5,6,7,8,9]]

    """
    l = len(x)
    step = math.ceil(l/nums)
    return [x[i:i+step] for i in range(0, l ,step)]


def getParser():
    parser = argparse.ArgumentParser(
        prog = str(__file__), 
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
        %prog cross corrs 
        @Author: xutingfeng@big.ac.cn 

        Version: 1.0
        Exmaple:
            1. for linear --method ols ; q is x while k is y; will conbiantion x1 with y....x_n with y and only use --key_cols or all if no --key_cols
            cross_corr.py -q Olink_v2.pkl -k cad.pkl -o cad_olink/cad_olink_linear.csv --method ols -t 5 --key_cols ldl_a
            2. for logistic --method logistic; q is x while k is y; will conbiantion x1 with y....x_n with y and only use --key_cols or all if no --key_cols
            cross_corr.py -q Olink_v2.pkl -k cad.pkl -o cad_olink/cad_olink_logistic.csv --method logistic -t 5 --key_cols cad
            3. for only pearson corr --method pearson; q and k will cal pearson corr --key_cols will use selected cols or all if no --key_cols
        """
        ),
    )
    # main params
    parser.add_argument("-q", "--query", dest="query", help="query file path", required=True)
    parser.add_argument("--query_cols", dest="query_cols", help="query cols to cal corrs, default all cols of query", required=False, nargs="+", default=[])
    # parser.add_argument("query_cat_cols", dest="query_cat_cols", help="cat cols in query file", required=False, nargs="+", default=[])

    parser.add_argument("-k", "--key", dest="key", help="key file path", required=True)
    parser.add_argument("--key_cols", dest="key_cols", help="key cols to cal corrs, default all cols of key", required=False, nargs="+", default=[])

    parser.add_argument("-o", "--output", dest = "output", help = "outpu file name", required=True)
    parser.add_argument("-t", "--threads", dest="threads", help="processes of this ", default=5, type=int, required=False)
    parser.add_argument("--adjust", dest="adjust", help="regress_out_confounding", action="store_true", required=False)

    parser.add_argument("--cond", dest="cond_path", help="confounding file path, should be as same as q and k and used with --method linear or logistic", required=False)
    parser.add_argument("--cond_cols", dest= "cond_cols", help="confounding cols, should be in cond_path files", required=False, nargs="+", default=[])
    parser.add_argument(
        "--cat_cond_cols",
        dest="cat_cond_cols",
        help="cat cols in cond file",
        required=False,
        nargs="+",
        default=[],
    )

    parser.add_argument("-m", "--method", dest="method", default="auto", required=False, choices=["glm", "ols", "logistic", "auto"]) # may supported for multiple method
    # parser.add_argument("--lowmem", action="store_true", dest="lowmem", help="low memory for cal")
    parser.add_argument("--norm_x", dest="norm_x",default=None, help="norm x before cal corrs, supported int or zscore", required=False, choices=["int", "zscore", None])
    parser.add_argument("--verbose", action="store_true", dest="verbose", help="print verbose output")

    return parser


def parse_input_data(
    query_path,
    key_path,
    query_cols=None,
    key_cols=None,
    cond_path=None,
    cond_cols=None,
    cat_cond_cols=None,
):

    query_df = load_data(query_path)
    key_df = load_data(key_path)
    cond_df = load_data(cond_path) if cond_path else None

    # eid may be the index name and not in the first columns in pkl
    if query_df.index.name =="eid":
        query_df.reset_index(inplace=True)
    if key_df.index.name =="eid":
        key_df.reset_index(inplace=True)
    if cond_df is not None and cond_df.index.name =="eid":
        cond_df.reset_index(inplace=True)

    columns_of_query = query_df.columns
    columns_of_key = key_df.columns
    columns_of_cond = cond_df.columns if cond_df is not None else None

    if query_cols is None:
        query_cols = []
    if key_cols is None:
        key_cols = []
    if cond_cols is None:
        cond_cols = []

    assert (columns_of_query[0] == columns_of_key[0]), "query and key should have the same first columns, but is " + columns_of_query[0] + " and " + columns_of_key[0] + " and " + columns_of_cond[0] if columns_of_cond is not None else "" + " respectively"
    if columns_of_cond is not None:
        assert (columns_of_query[0] == columns_of_cond[0]), "query and key and  cond should have the same first columns"
    merge_on = columns_of_query[0]

    # set merge_on as str
    query_df[merge_on] = query_df[merge_on].astype(str)
    key_df[merge_on] = key_df[merge_on].astype(str)
    if cond_df is not None:
        cond_df[merge_on] = cond_df[merge_on].astype(str)

    # drop merge_on
    columns_of_query = columns_of_query[columns_of_query != merge_on]
    columns_of_key = columns_of_key[columns_of_key != merge_on]
    columns_of_cond = columns_of_cond[columns_of_cond != merge_on] if columns_of_cond is not None else None

    query_cols = columns_of_query[columns_of_query.str.contains('|'.join(query_cols))].tolist()
    key_cols = columns_of_key[columns_of_key.str.contains('|'.join(key_cols))].tolist()
    cond_cols = columns_of_cond[columns_of_cond.str.contains('|'.join(cond_cols))].tolist() if columns_of_cond is not None else None

    # TODO: check if query_cols, key_cols, cond_cols are empty or some col are same

    # keep only the cols in query_cols, key_cols, cond_cols
    query_df = query_df[[merge_on] + query_cols]
    key_df = key_df[[merge_on] + key_cols]
    if cond_df is not None:
        cond_df = cond_df[[merge_on] + cond_cols]

    #  query col > key col > cond_cols ; to keep query col not in key col and cond col ; key col not in cond col
    for q_col in query_cols:
        if q_col in key_cols:
            print(f"query col {q_col} is in key col, will drop it")
            key_cols.remove(q_col)
        if len(cond_cols) > 0:
            if q_col in cond_cols:
                raise ValueError(f"query col {q_col} is in cond col, should drop it or avoid this happen!!!!")

    for k_col in key_cols:
        if len(cond_cols) > 0:
            if k_col in cond_cols:
                raise ValueError(f"key col {k_col} is in cond col, should drop it or avoid this happen!!!!")

    # merge

    main_df = query_df.merge(key_df, on = merge_on, how="inner")
    if cond_df is not None:
        main_df = main_df.merge(cond_df, on = merge_on, how="inner")

    msg = f"query_col have {len(query_cols)} cols and first 5 cols are {query_cols[:5]}\nkey_col have {len(key_cols)} cols and first 5 cols are {key_cols[:5]}\n" 

    # filter cov
    for col in query_cols:
        try:
            main_df[col] = main_df[col].astype(float)
        except:
            main_df.drop(col, axis=1, inplace=True)
            query_cols.remove(col)

    for col in key_cols:
        try:
            main_df[col] = main_df[col].astype(float)
        except:
            main_df.drop(col, axis=1, inplace=True)
            key_cols.remove(col)
    if cond_cols:
        for col in cond_cols:
            try:
                main_df[col] = main_df[col].astype(float)
            except:
                print
                main_df.drop(col, axis=1, inplace=True)
                cond_cols.remove(col)

    if len(cond_cols) > 0:
        cat_cond_cols = [col for col in cond_cols if col in cat_cond_cols]
        cond_cols = [col for col in cond_cols if col not in cat_cond_cols]
        if len(cat_cond_cols) > 0:
            msg += f"cat_cond_col have {len(cat_cond_cols)} cols and first 5 cols are {cat_cond_cols[:5]}\n"
        if len(cond_cols) > 0:
            msg += f"cond_col have {len(cond_cols)} cols and first 5 cols are {cond_cols[:5]}\n"

    msg += f"Total shape is {main_df.shape}"
    print(msg)

    return main_df, {"query_cols":query_cols, "key_cols":key_cols, "cond_cols":cond_cols, "cat_cond_cols":cat_cond_cols}


if __name__ == "__main__":
    # input , currently only supported for two files
    parser = getParser()
    args = parser.parse_args()

    query_path = args.query
    key_path = args.key
    output = args.output
    threads = args.threads
    norm_x = args.norm_x
    # index_cols = None if len(args.index_on) ==0 else args.index_on
    method = args.method
    # lowmem = args.lowmem
    adjust = args.adjust
    cond_path = args.cond_path
    # cond_cols_used = args.cond_cols # [Age, Sex]....
    key_cols_used = args.key_cols
    verbose = args.verbose
    timing = Timing()
    # # confilict params check
    # if method not in ["logistic", "linear", "glm"]:
    #     if cond_path or len(cond_cols_used) > 0:
    #         raise ValueError("confounding only supported for logistic and linear method")

    # read data
    cond_cols = args.cond_cols + args.cat_cond_cols
    cat_cond_cols = args.cat_cond_cols
    main_df, col_dict = parse_input_data(
        query_path=query_path,
        key_path=key_path,
        query_cols=args.query_cols,
        key_cols=args.key_cols,
        cond_path=cond_path,
        cond_cols=cond_cols,
        cat_cond_cols=cat_cond_cols,
    )
    # print(main_df)

    corr_results_df = cal_corr_v2(
        df=main_df,
        x=col_dict["query_cols"],
        y=col_dict["key_cols"],
        # cofounder=col_dict["cond_cols"],
        cov=col_dict["cond_cols"],
        cat_cov=col_dict["cat_cond_cols"],
        adjust=adjust,
        norm_x=norm_x,
        model_type=method,
        threads=threads,
        verbose=verbose,
    )

    if Path(output).parent.exists() is False:
        Path(output).parent.mkdir(parents=True, exist_ok=True)

    corr_results_df.dropna(how="all", inplace=True)
    if output.endswith(".gz"):
        corr_results_df.to_csv(output, compression="gzip", index=False, na_rep="NA", sep="\t")
    else:
        corr_results_df.to_csv(output, index=False, na_rep="NA", sep= "\t")

    print(f"总共消耗{timing():.2f}s")

    # if lowmem: # save to local tmp file and read while running
    #     parts_df = []
    #     tmp_dir = Path(output).parent
    #     print(f"--lowmem ， 采用低内存模式，将会保存中间文件到本地，可能会占用大量磁盘空间，保存在该路径下：{str(tmp_dir)}")

    #     for idx, part_cols in enumerate(average_list(query_cols, threads)):
    #         tmp_save_file = tmp_dir/f"tmp_part_{idx}.pkl"
    #         if cond_path: # save with cond_cols
    #             main_df[part_cols + key_cols + cond_cols].to_pickle(str(tmp_save_file))
    #         else:
    #             main_df[part_cols + key_cols].to_pickle(str(tmp_save_file))
    #         parts_df.append(tmp_save_file)
    # else:
    #     if cond_path:
    #         parts_df = [main_df[parts_col + key_cols + cond_cols].copy() for parts_col in average_list(query_cols, threads)]
    #     else:
    #         parts_df = [main_df[parts_col + key_cols].copy() for parts_col in average_list(query_cols, threads)]  # [part1_df, part2_df, ....]

    # del query_df  # clear for memory
    # del main_df # clear for memory

    # # TODO: support for regression method
    # if cond_path:
    #     cal_corrs_multiprocess = partial(cross_corrs, key_cols = key_cols, method=method, cond_cols=cond_cols)
    # else:
    #     cal_corrs_multiprocess = partial(cross_corrs, key_cols = key_cols, method=method)

    # with Pool(threads) as p:
    #     res = p.map(cal_corrs_multiprocess, parts_df)

    # corr_results_df = pd.concat(res).reset_index(drop=True)
    # corr_results_df = generate_multipletests_result(corr_results_df, pvalue_col="pvalue", alpha=0.05, method="fdr_bh")
    # if lowmem:
    #     for tmp_path in parts_df:
    #         os.remove(tmp_path)

    # save files
