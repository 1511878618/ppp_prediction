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


# rank_INT end 


# def cal_pearsonr(x, y):
#     try:
#         r, p = pearsonr(x, y)
#     except:
#         r, p = None, None 
#     return {"r":r, "pvalue":p}
    

# def logistic_regression(data, x, y, confounding=None):
#     # Define the independent variables (X) and the dependent variable (y)
#     if confounding:
#         if not isinstance(confounding, list):
#             confounding = [confounding]


#     x_counfound = [x] + confounding if confounding else [x]

#     # drop na 
#     # print(x, y, confounding)
#     # data = data.dropna(subset=x_counfound + [y], how="any").reset_index(drop=True)
#     used_cols = x_counfound + [y]
#     data = data[used_cols].dropna(how="any").reset_index(drop=True)
#     print(f"x is {x} and y is {y}, confounding is {confounding}; after dropna, keep {data.shape[0]} rows in analysis.")

#     if confounding:
#         print(f"x is {x} and y is {y} with conditional: {','.join(confounding)} and shape is {data.shape}")
#     else: 
#         print(f"x is {x} and y is {y} and shape is {data.shape}")
#     N = data.shape[0]
#     N_case = data[y].sum()
#     N_control = N - N_case

#     X = data[x_counfound]
#     y = data[y]

#     # Add a constant term to the independent variables
#     X = sm.add_constant(X)

#     # Fit the logistic regression model
#     model = sm.Logit(y, X)
#     try:
#         result = model.fit()
#         model.fit_regularized()

#         # Get the beta coefficients
#         beta = result.params

#         # Get the p-values of the coefficients
#         p_values = result.pvalues

#         return {"beta": beta[x], "pvalue":p_values[x], "N":N, "N_case" : N_case, "N_control" : N_control}
#     except:
#         return {"beta": None, "pvalue":None, "N":None, "N_case":None, "N_control":None}

# def linear(data, x, y, confounding=None):
#     # Define the independent variables (X) and the dependent variable (y)
#     if confounding:
#         if not isinstance(confounding, list):
#             confounding = [confounding]
    
#     x_counfound = [x] + confounding if confounding else [x]

#     # drop na 
#     data = data.dropna(subset=x_counfound + [y], how="any").reset_index(drop=True)
#     print(f"x is {x} and y is {y}, confounding is {confounding}; after dropna, keep {data.shape[0]} rows in analysis.")
#     N = data.shape[0]
#     X = data[x_counfound]
#     y = data[y]

#     # Add a constant term to the independent variables
#     X = sm.add_constant(X)

#     # Fit the logistic regression model
#     model = sm.OLS(y, X, )


#     result = model.fit()

#     # Get the beta coefficients
#     beta = result.params

#     # Get the p-values of the coefficients
#     p_values = result.pvalues

#     return {"beta": beta[x], "pvalue":p_values[x], "N":N}
#     # except:
#     #     return {"beta": None, "pvalue":None}



# def cal_corrs(data, x, y, method, cond_cols=None):
#     if method == "pearson":
#         tmp_data = data[[x, y]].dropna()
#         return cal_pearsonr(tmp_data[x], tmp_data[y])
#     elif method == "logistic":
#         return logistic_regression(data, x, y, cond_cols)
#     elif method == "linear":
#         return linear(data, x, y, cond_cols)
#     else:
#         return NotImplementedError(f"method {method} not supported yet")



# def cross_corrs(main_df, query_cols,key_cols, method="pearson", cond_cols=None):
#     """
#     main_df cols is equal: [query_cols, key_cols]
#     so query_cols = main_df.columns - key_cols

#     will combination between query_cols and key_cols 

#     return: pd.DataFrame
    
#     """
#     res = []
#     if isinstance(main_df, str): # read tmp_part_file_path
#         main_df = read_data(main_df)

#     # query_cols = list(set(main_df.columns) - set(key_cols)) if cond_cols is None else list(set(main_df.columns) - set(key_cols) - set(cond_cols)) # main_df.columns - key_cols => query_cols 
    
#     for query_col in query_cols: # query 
#         for key_col in key_cols: # key

#             corr_dict = cal_corrs(data = main_df, x= query_col, y=key_col, method = method, cond_cols = cond_cols)
#             res.append({**{"query":query_col, "key":key_col} , **corr_dict})

#     return pd.DataFrame(res) 


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

    parser.add_argument("-k", "--key", dest="key", help="key file path", required=True)
    parser.add_argument("--key_cols", dest="key_cols", help="key cols to cal corrs, default all cols of key", required=False, nargs="+", default=[])

    parser.add_argument("-o", "--output", dest = "output", help = "outpu file name", required=True)
    parser.add_argument("-t", "--threads", dest="threads", help="processes of this ", default=5, type=int, required=False)
    parser.add_argument("--adjust", dest="adjust", help="regress_out_confounding", action="store_true", required=False)

    parser.add_argument("--cond", dest="cond_path", help="confounding file path, should be as same as q and k and used with --method linear or logistic", required=False)
    parser.add_argument("--cond_cols", dest= "cond_cols", help="confounding cols, should be in cond_path files", required=False, nargs="+", default=[])

    parser.add_argument("-m", "--method", dest="method", default="pearson", required=False, choices=["glm", "ols", "logistic"]) # may supported for multiple method
    # parser.add_argument("--lowmem", action="store_true", dest="lowmem", help="low memory for cal")
    parser.add_argument("--norm_x", dest="norm_x",default=None, help="norm x before cal corrs, supported int or zscore", required=False, choices=["int", "zscore", None])
    parser.add_argument("--verbose", action="store_true", dest="verbose", help="print verbose output")

    return parser


def parse_input_data(query_path, key_path, query_cols=None,  key_cols=None, cond_path=None, cond_cols=None):

    query_df = read_data(query_path)
    key_df = read_data(key_path)
    cond_df = read_data(cond_path) if cond_path else None

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

    main_df = query_df.merge(key_df, on = merge_on, how="inner")
    if cond_df is not None:
        main_df = main_df.merge(cond_df, on = merge_on, how="inner")
    print(f"query_col have {len(query_cols)} cols and first 5 cols are {query_cols[:5]}\nkey_col have {len(key_cols)} cols and first 5 cols are {key_cols[:5]}\ncond_col have {len(cond_cols)} cols and first 5 cols are {cond_cols[:5]}\nTotal shape is {main_df.shape}")


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
                main_df.drop(col, axis=1, inplace=True)
                cond_cols.remove(col)



    return main_df, {"query_cols":query_cols, "key_cols":key_cols, "cond_cols":cond_cols}



        

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
    cond_cols_used = args.cond_cols # [Age, Sex]....
    key_cols_used = args.key_cols
    verbose = args.verbose
    timing = Timing()
    # # confilict params check 
    # if method not in ["logistic", "linear", "glm"]:
    #     if cond_path or len(cond_cols_used) > 0:
    #         raise ValueError("confounding only supported for logistic and linear method")

    # read data
    main_df, col_dict = parse_input_data(
        query_path = query_path, 
        key_path = key_path, 
        query_cols = args.query_cols, 
        key_cols = args.key_cols, 
        cond_path = cond_path, 
        cond_cols = cond_cols_used
    )

    corr_results_df = cal_corr_v2(
        df=main_df,
        x=col_dict["query_cols"],
        y=col_dict["key_cols"],
        cofounder=col_dict["cond_cols"],
        adjust=adjust,
        norm_x = norm_x,
        model_type=method,
        threads=threads,
        verbose=verbose

    )

    if Path(output).parent.exists() is False:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
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
