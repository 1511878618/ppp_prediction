import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import logging

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
        default=5,
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

