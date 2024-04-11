#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Description:       :
@Date     :2024/04/06 15:59:28
@Author      :Tingfeng Xu
@version      :1.0


# step1 bootstrap training to get the dist of weights 

# step2 Sensitivity analysis with different threshold by mean weights 

# step3 use the best cutoff to get the final proteins 


"""

from multiprocessing import Pool
import argparse
from pathlib import Path
import textwrap
import json
import matplotlib.pyplot as plt
import logging
import sys
import json
import pickle
from ppp_prediction.model import fit_best_model, fit_best_model_bootstrap, EnsembleModel, lasso_select_model
from ppp_prediction.corr import cal_binary_metrics_bootstrap
from ppp_prediction.utils import DataFramePretty

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s\n%(message)s",
    stream=sys.stdout,
)


def args_parse():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            %prog is ...
            @Author: xutingfeng@big.ac.cn
            Version: 1.0
            input json should:
            {
            "combination1": {
                "features": ["feature1", "feature2"],
                "label": "label1"
            },
            "combination2": {
                "features": ["feature3", "feature4"],
                "label": "label2"
                },
                ...

            }
            """
        ),
    )
    parser.add_argument("--train", required=False, help="input train file")
    parser.add_argument("--test", required=False, help="input test file")
    parser.add_argument("--output", required=False, help="output file")
    parser.add_argument(
        "--json", required=False, help="json file for  combination of input"
    )
    parser.add_argument(
        "-n",
        "--n-bootstrap",
        default=1,
        type=int,
        help="n bootstrap, if n>1 will use bootstrap to build ensemble model",
    )
    parser.add_argument(
        "-t",
        "--threads",
        default=1,
        type=int,
        help="threads for bootstrap ,if n=1 this will not work",
    )
    parser.add_argument(
        "--cv",
        default=3,
        type=int,
        help="cv for fit_best_model, default is 3",
    )
    parser.add_argument(
        "--debug",
        default=False,
        # type=bool,
        action="store_true",
    )
    return parser.parse_args()


def check_combination_json_file(json_file, *dfs):
    """
    check combination json file and columns in dataframes
    """
    error_message_nums = 0
    for key, value in json_file.items():
        features = value["features"]
        label = value["label"]
        for df in dfs:
            columns = df.columns
            for feature in features:
                if feature not in columns:
                    logging.error(f"feature {feature} not in columns {columns}")
                    error_message_nums += 1
            if label not in columns:
                logging.error(f"label {label} not in columns {columns}")
                error_message_nums += 1
    if error_message_nums > 0:
        raise ValueError(
            f"error in json file and columns, with error message nums: {error_message_nums}, this means not all features and labels in json file are in dataframes columns, plz check"
        )


import pandas as pd


def load_df(file):

    if file.endswith("tsv") or file.endswith(".tsv.gz"):
        return pd.read_csv(file, sep="\t")
    elif file.endswith(".pkl"):
        return pd.read_pickle(file)
    else:
        try:
            return pd.read_csv(file)
        except:
            raise ValueError(f"file {file} not supported")
    
    

if __name__ == "__main__":
    args = args_parse()


    output_file = args.output
    json_file = args.json
    n_bootstrap = args.n_bootstrap
    threads = args.threads

    if args.debug:
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import load_breast_cancer

        breast_cancer = load_breast_cancer(as_frame=True)
        X = breast_cancer.data
        y = breast_cancer.target.astype(float)
        df = pd.concat([X, y], axis=1)

        train_file, test_file = train_test_split(df, test_size=0.2, random_state=42)

        features = df.columns[:-1].tolist()
        target = df.columns[-1]
        print(f"debug open, use breast_cancer data to debug lasso_select_model, will save to ./debug")
        lasso_select_model(
            train_df=train_file,
            test_df=test_file,
            features=features,
            label=target,
            cv=2,
            threads=1,
            n_bootstrap=4,
            save_dir="./debug",
            name="debug",
        )


    else:
        train_file = load_df(args.train)
        test_file = load_df(args.test)

    combination_json = json.load(open(json_file))
    logging.info(f"combination used is {combination_json.keys()}")

    check_combination_json_file(combination_json, train_file, test_file)

    # Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    Regression_model_result_dict = {}
    for key, value in combination_json.items():
        logging.info(f"start to fit model for combination {key}")
        current_save_path = f"{output_file}/{key}"
        lasso_select_model(
            train_df=train_file,
            test_df=test_file,
            features=value["features"],
            label=value["label"],
            cv=args.cv,
            threads=threads,
            n_bootstrap=n_bootstrap,
            save_dir=current_save_path,
            name=key,
        )


