#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Description:       :
@Date     :2024/04/06 15:59:28
@Author      :Tingfeng Xu
@version      :1.0
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
from ppp_prediction.model import fit_best_model, fit_best_model_bootstrap, EnsembleModel
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
    parser.add_argument("--train", required=True, help="input train file")
    parser.add_argument("--test", required=False, help="input test file")
    parser.add_argument("--output", required=True, help="output file")
    parser.add_argument(
        "--json", required=True, help="json file for  combination of input"
    )
    parser.add_argument(
        "-m",
        "--method",
        required=False,
        default="Lasso",
        help="method list for fit_best_model, default is ['Lasso', 'ElasticNet', 'Logistic']",
    )


    parser.add_argument(
        "--cv",
        default=5,
        type=int,
        help="cv for fit_best_model, default is 3",
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

    train_file = load_df(args.train)
    test_file = load_df(args.test)
    output_file = args.output
    json_file = args.json
    method = args.method

    combination_json = json.load(open(json_file))
    logging.info(f"combination used is {combination_json.keys()}")

    check_combination_json_file(combination_json, train_file, test_file)

    # Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    Regression_model_result_dict = {}
    for key, value in combination_json.items():
        logging.info(f"start to fit model for combination {key}")
        current_save_pkl_path = f"{output_file}/{key}.pkl"
        Path(output_file).mkdir(parents=True, exist_ok=True)
        if Path(current_save_pkl_path).exists():
            logging.info(
                f"{key} is fited before at {current_save_pkl_path}, skip this!"
            )
            continue

        features = value["features"]
        label = value["label"]


        (
            model,
            train_metrics,
            test_metrics,
            train_imputed_data,
            test_imputed_data,
            best_models,
        ) = fit_best_model(
            train_df=train_file,
            test_df=test_file,
            X_var=features,
            y_var=label,
            method_list=[method],
            cv=args.cv,
        )
        # plot data 
        

        test_metrics = cal_binary_metrics_bootstrap(
            test_imputed_data[label],
            test_imputed_data[f"{label}_pred"],
            ci_kwargs=dict(n_resamples=1000),
        )
        all_obj = {
            "model": model,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            # "train_data": train_imputed_data,
            # "test_data": test_imputed_data,
        }

        Regression_model_result_dict[key] = test_metrics

        try:
            DataFramePretty(pd.Series(test_metrics).to_frame()).show()
        except:
            pass

        pickle.dump(all_obj, open(current_save_pkl_path, "wb"))
        print(train_metrics)
        print(test_metrics)

    Regression_model_result_df = pd.DataFrame(Regression_model_result_dict).T
    DataFramePretty(Regression_model_result_df).show()
    Regression_model_result_df.to_csv(
        f"{output_file}/Regression_model_result_df.tsv", sep="\t"
    )
