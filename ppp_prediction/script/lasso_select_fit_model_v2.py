#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Description:       :
@Date     :2024/10/28 18:59:28
@Author      :Tingfeng Xu
@version      :2.0


# step1 bootstrap training to get the dist of weights 

# step2 Sensitivity analysis with different threshold by mean weights 

# step3 for each threshold use the cutoff to fit model by the model list 


"""


import argparse
from pathlib import Path
import textwrap

import logging
import sys
import json

import pandas as pd
from ppp_prediction.utils import load_data
from ppp_prediction.model_v2.lasso_select_fit import BootstrapLassoSelectModelAndFit
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

            }
            """
        ),
    )
    parser.add_argument("--train_dir", required=False, help="input train file")
    parser.add_argument("--test_dir", required=False, help="input test file")
    parser.add_argument("--output", required=False, help="output folder")
    parser.add_argument("--label_dir", required=False, help="label file")
    parser.add_argument("--label", required=False, help="label name")

    # optional
    
    parser.add_argument(
        "-n",
        "--n-bootstrap",
        default=100,
        type=int,
        help="n bootstrap, if n>1 will use bootstrap to build ensemble model",
    )
    parser.add_argument(
        "-t",
        "--threads",
        default=5,
        type=int,
        help="threads for bootstrap ,if n=1 this will not work",
    )
    parser.add_argument(
        "--cv",
        default=5,
        type=int,
        help="cv for fit_best_model, default is 3",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="debug mode",
    )
    parser.add_argument(
        "--bootstrap_engine",
        default="sklearn",
        type=str,
        help="bootstrap engine, cuml or sklearn",
    )

        

    return parser.parse_args()


    

if __name__ == "__main__":
    args = args_parse()


    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    output_file = Path(args.output)
    label_dir = Path(args.label_dir)
    label = args.label
    n_bootstrap = args.n_bootstrap
    threads = args.threads
    cv = args.cv



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
        BootstrapLassoSelectModelAndFit(
            train = train_file,
            test = test_file,
            xvar = features,
            label = target,
            cv=2,
            threads=1,
            n_bootstrap=4,
            save_dir="./debug",
            name="debug",
        )
        sys.exit(0)

    else:
        train_file = load_data(train_dir)
        test_file = load_data(test_dir)
        label_file = load_data(label_dir)

        # merge label to train_file
        if 'eid' not in train_file.columns:
            # train_file['eid'] = train_file.index
            raise ValueError("train_file should have 'eid' column")
        if 'eid' not in test_file.columns:
            # test_file['eid'] = test_file.index
            raise ValueError("test_file should have 'eid' column")
        if 'eid' not in label_file.columns:
            # label_file['eid'] = label_file.index
            raise ValueError("label_file should have 'eid' column")
        
        features = [col for col in train_file.columns if col not in ['eid']]

        for feature in features:
            if feature not in test_file.columns:
                raise ValueError(f"feature {feature} not in test_file")
        
            if feature in label_file.columns:
                raise ValueError(f"feature {feature} in label_file")
            
        


        train_file = train_file.merge(label_file, on= 'eid', how='inner')
        test_file = test_file.merge(label_file, on= 'eid', how='inner')
        

        print(f"train_file shape: {train_file.shape} with label {label} and {train_file[label].value_counts()}")
        print(f"test_file shape: {test_file.shape} with label {label} and {test_file[label].value_counts()}")


        method_trained_dict, compare_df = BootstrapLassoSelectModelAndFit(
            train_file,
            features,
            label,
            test=test_file,
            methods=["Lasso", "lightgbm", "xgboost"],
            threads=threads,
            save_dir=output_file,
            bootstrap_engine=args.bootstrap_engine,
            bootstrap_n_resample=n_bootstrap,
        )

