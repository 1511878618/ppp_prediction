#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Description:       :
@Date     :2024/04/06 15:59:28
@Author      :Tingfeng Xu
@version      :1.0
"""

import argparse
from pathlib import Path
import textwrap
import json
import logging
import sys
import json
from ppp_prediction.corr import cal_corr
import pandas as pd
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s\n%(message)s",
    stream=sys.stdout,
)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def args_parse():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            %prog is ...
            @Author: xutingfeng@big.ac.cn
            Version: 1.0
            --json xxx.json --whole_file xxx.pkl
            input json should:
            {
            "outputName1": {
                features: ["feature1", "feature2"],
                label: "label1",
                cofounder: ["cofounder1", "cofounder2"],
                }
                ...
            }
            or Not implemented yet
            by --feature xxx.pkl --label xxx.pkl --cofounder xxx.pkl --out outname with first column as index  (csv is ok, but make sure first is index)
            """
        ),
    )
    parser.add_argument("--whole_file", required=False, help="input whole file")
    parser.add_argument("--json", required=False, help="input json file")
    parser.add_argument("-y", required=False, help="replace the old file if it exists")
    # parser.add_argument("--feature", required=False, help="input feature file")
    # parser.add_argument("--label", required=False, help="input label file")
    # parser.add_argument("--cofounder", required=False, help="input cofounder file")
    parser.add_argument("--out", required=False, help="output file")

    return parser.parse_args()


from functools import reduce


def check_combination_json_file(json_file, *dfs):
    """
    check combination json file and columns in dataframes
    """
    error_message_nums = 0
    for key, value in json_file.items():
        features = value["features"]
        label = value["label"]
        cofounder = value["cofounders"]
        to_checek = reduce(
            lambda x, y: x + y, [k for k in value.values() if isinstance(k, list)]
        )
        logging.info(f"to check {len(to_checek)} columns from json in dataframes")
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


def parse_file_and_json(whole_file, json_file):
    """
    parse whole file and json file
    """
    combination_json = json.load(open(json_file))
    whole_file = load_df(whole_file)
    check_combination_json_file(combination_json, whole_file)

    return whole_file, combination_json


if __name__ == "__main__":
    args = args_parse()
    association_analysis_output_path = args.out
    Path(association_analysis_output_path).mkdir(parents=True, exist_ok=True)
    if args.whole_file and args.json:
        logging.info("whole file and json file is passed will be used")
        whole_file, json_file = parse_file_and_json(args.whole_file, args.json)

    else:
        raise ValueError(
            "plz pass whole_file and json or feature, label, cofounder and out"
        )

    res = []
    for k, v in json_file.items():
        features = v["features"]
        label = v["label"]
        cofounder = v["cofounders"]
        current_output_dir = f"{association_analysis_output_path}/{label}.csv"
        if Path(current_output_dir).exists():
            if args.y:
                logging.info(f"file {current_output_dir} exists, will be replaced")
            else:
                logging.info(f"file {current_output_dir} exists, will be skipped")
                continue
        logging.info(f"start to calculate {label}")

        single_association_proteins_result_df = cal_corr(
            whole_file, features, y=label, cofounders=cofounder, model_type="logit"
        )

        single_association_proteins_result_df.to_csv(current_output_dir, index=False)
        logging.info(f"save {label}.csv")
        res.append(single_association_proteins_result_df)

    if len(res) > 1:
        res_df = pd.concat(res)
        res_df.to_csv(f"{association_analysis_output_path}/all.csv", index=False)
