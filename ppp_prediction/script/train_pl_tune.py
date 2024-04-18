#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Description:       :
@Date     :2023/11/16 19:52:09
@Author      :Tingfeng Xu
@version      :2.0
"""

import torchvision
from pytorch_lightning import Trainer, seed_everything

seed_everything(42)

from pytorch_lightning import LightningDataModule
import torch
from torch.nn import functional as F
import torch
import torchmetrics

from torch import nn
import pandas as pd
from pathlib import Path
import numpy as np
import warnings
torchvision.disable_beta_transforms_warning()

import argparse
import textwrap
import warnings

import torch
from pytorch_lightning import Trainer, seed_everything



from pytorch_lightning.loggers import TensorBoardLogger
from ray import train, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback,
)
import os 
import math 
from ray.train import RunConfig

from pytorch_lightning import Callback
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from torch.utils.data import (
    DataLoader,
    SequentialSampler,
    RandomSampler,
    WeightedRandomSampler,
    Dataset,
)
import numpy as np

from tqdm.rich import tqdm
import numpy as np

from statsmodels.stats.multitest import multipletests



from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
)
from ppp_prediction.dl.model import LinearTransformer
from ppp_prediction.dl.pl_model import LinearTransformerPL
from ppp_prediction.dl.dataset import TableDataset, TableDatasetModule


import os
from torch import nn

import torch
import pytorch_lightning as pl

import torch.nn as nn
from collections import defaultdict

class HistoryCallback(Callback):

    def __init__(self, rootPath, **kwargs) -> None:
        super().__init__()

        self.log_dir = Path(rootPath) / "history/history.csv"
        self.log_dir.parent.mkdir(parents=True, exist_ok=True)


    def on_train_epoch_end(self, trainer, module):
        history = pd.DataFrame(module.history)
        history.to_csv(str(self.log_dir), index=False ) 

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        # return super().on_test_epoch_end(trainer, pl_module)
        history = pd.DataFrame(pl_module.history)
        history.to_csv(str(self.log_dir), index=False )


seed_everything(42, workers=True)  # seed for reproducibility
warnings.filterwarnings("ignore")



def getParser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            %prog is ...
            @Author: xutingfeng@big.ac.cn
            Version: 1.1

    
                ...
            """
        ),
    )
    parser.add_argument("--train", required=True, help="input train file")
    parser.add_argument("--test", required=False, help="input test file")
    parser.add_argument("--output", required=True, help="output file")
    parser.add_argument("--resources", default=[], nargs="+", help="resources, default is [2, 4 , 0.5], 2 threads 4 cpu and 0.5 gpu",)
    parser.add_argument("--config", default="config.yaml", help="config file")



    return parser


if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()

    train_imputed = pd.read_csv(args.train) if args.train.endswith(".csv") else pd.read_pickle(args.train)
    test_imputed = pd.read_csv(args.test) if args.test.endswith(".csv") else pd.read_pickle(args.test)
    resource = args.resources if len(args.resources) > 0 else [2, 4, 0.5]
    parallel_trile, cpu_nums, gpu_nums = resource
    parallel_trile = int(parallel_trile)
    cpu_nums = int(cpu_nums)
    gpu_nums = float(gpu_nums)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # The maximum training epochs
    num_epochs = 20

    # Number of sampls from parameter space
    num_samples = 100

    # params  
    proteomics = test_imputed.columns[test_imputed.columns.tolist().index("C3") :].tolist()  # currently only for DL 
    risk_factors = [
    "age",
    "sex",
    "ldl_a",
    "hdl_a",
    "tc_a",
    "tg_a",
    "sbp_a",
    "BMI",
    "smoking",
    "prevalent_diabetes",
]

    LinearTransformerPL_search_space = {
        "features_dict": {"proteomics": proteomics},
        "covariates_dict": tune.choice([{"risk_factors": risk_factors}, None]),
        "d_ff": tune.choice([64, 128, 256, 512]),
        "num_classes": 2,
        "num_layers": tune.choice([1, 2, 3, 4, 5]),
        "dropout": tune.uniform(0.1, 0.5),
        "lr": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-4, 1e-2),
        "weight": tune.choice([[1, 1], [0.1, 1], [0.1, 10], [0.1, 100]]),
        "batch_size": tune.choice([64, 256]),
    }

    ###### ray tune########

    from ray.train.lightning import (
        RayDDPStrategy,
        RayLightningEnvironment,
        RayTrainReportCallback,
        prepare_trainer,
    )
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.train import RunConfig, ScalingConfig, CheckpointConfig
    from ray.train.torch import TorchTrainer


    def train_func(config):
        features_key = list(config["features_dict"].keys())[0]
        covariates_key = (
            list(config["covariates_dict"].keys())[0] if config["covariates_dict"] else None
        )

        dataset = TableDatasetModule(
            train=train_imputed,
            test=test_imputed,
            features=config["features_dict"][features_key],
            covariates=(
                config["covariates_dict"][covariates_key]
                if config["covariates_dict"]
                else None
            ),
            label=["incident_cad"],
            num_classes=2,
            batch_size=config["batch_size"],
        )

        model = LinearTransformerPL(**config)
        trainer = Trainer(
            devices="auto",
            strategy=RayDDPStrategy(),
            callbacks=[RayTrainReportCallback()],
            plugins=[RayLightningEnvironment()],
            enable_progress_bar=True,
        )
        trainer = prepare_trainer(trainer)
        trainer.fit(model, dataset)




    search_space = LinearTransformerPL_search_space


    # The maximum training epochs
    num_epochs = 20

    # Number of sampls from parameter space
    num_samples = 50
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)


    scaling_config = ScalingConfig(
        num_workers=parallel_trile, use_gpu=True, resources_per_worker={"CPU": cpu_nums, "GPU":gpu_nums}
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="ptl/val_auc",
            checkpoint_score_order="max",
        ),
    )

    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )


    def tune_asha(num_samples=10):
        scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

        tuner = tune.Tuner(
            ray_trainer,
            param_space={"train_loop_config": search_space},
            tune_config=tune.TuneConfig(
                metric="ptl/val_auc",
                mode="max",
                num_samples=num_samples,
                scheduler=scheduler,
            ),
        )
        return tuner.fit()


    results = tune_asha(num_samples=num_samples)


    # results.get_best_result("ptl/val_auc")

    best_result = results.get_best_result("ptl/val_auc")
    best_params = best_result.config
    best_result_epoch_dir = (
        best_result.get_best_checkpoint("ptl/val_auc", "max").path + "/checkpoint.ckpt"
    )
    best_model_state = torch.load(best_result_epoch_dir)
    best_model = LinearTransformerPL(**best_params["train_loop_config"])
    best_model.load_state_dict(best_model_state["state_dict"])
    
    ## save score 
    test_imputed = best_model.predict_df(test_imputed)
    train_imputed = best_model.predict_df(train_imputed)

    train_imputed[['eid', 'pred']].to_csv(f"{args.output}/train_score.csv", index=False)
    test_imputed[['eid', 'pred']].to_csv(f"{args.output}/test_score.csv", index=False)  
    ## cal metrics 
    from ppp_prediction.corr import cal_binary_metrics_bootstrap


    ## 
    train_metrics = cal_binary_metrics_bootstrap(train_imputed['incident_cad'], train_imputed['pred'], ci_kwargs={"n_resamples":1000})
    test_metrics = cal_binary_metrics_bootstrap(test_imputed['incident_cad'], test_imputed['pred'], ci_kwargs={"n_resamples":1000})

    import pickle 
    with open(f"{args.output}/train_metrics.pkl", "wb") as f:
        pickle.dump(train_metrics, f)
    with open(f"{args.output}/test_metrics.pkl", "wb") as f:
        pickle.dump(test_metrics, f)
