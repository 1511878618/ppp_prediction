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


def pridict(model, predict_loader, device="cuda:0"):
    model.eval()
    model.to(device)
    res = []
    with torch.no_grad():
       
        for x, y in tqdm(predict_loader, desc="predicting", total=len(predict_loader)):
            y_hat = model(x.to(device))
            if model.task_type == "bt" and model.num_classes == 2:
                y_hat = torch.softmax(y_hat, dim=1)[:, 1].detach().cpu().numpy()
            elif model.task_type == "qt":
                y_hat = y_hat.detach().cpu().numpy()
            elif model.task_type == "bt" and model.num_classes > 2:
                y_hat = torch.softmax(y_hat, dim=1).detach().cpu().numpy()

            if isinstance(y[0], dict):
                for i in range(len(y)):
                    y[i]["y_hat"] = y_hat[i]
                    res.append(y[i])
            else:
                for i in range(len(y)):
                    res.append({"y": y[i], "y_hat": y_hat[i]})
    return res 




class TableDataset(Dataset):
    def __init__(self, df, features: list, label: list, num_classes=2, y_type="bt"):
        super(Dataset, self).__init__()
        assert isinstance(df, pd.DataFrame)
        assert isinstance(features, list)
        assert isinstance(label, list)

        for feature in features + label:
            assert feature in df.columns

        self.df = df.dropna(subset=features + label)
        assert len(self.df) > 0
        self.features = features
        self.label = label
        self.num_classes = num_classes
        self.y_type = y_type
        self._init_dataset()

    def _init_dataset(self):
        X = torch.tensor(self.df[self.features].values).float()

        y = torch.tensor(self.df[self.label].values)
        if (self.num_classes != len(self.label)) and self.y_type == "bt":
            y = F.one_hot(
                torch.tensor(y).long(), num_classes=self.num_classes
            ).squeeze()

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DatasetModule(LightningDataModule):
    def __init__(
        self,
        train,
        test,
        batch_size=32,
        features: list = None,
        label: list = None,
        num_classes=2,
        y_type="bt",
        num_workers=4,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.features = features
        self.label = label
        self.num_classes = num_classes
        self.y_type = y_type
        self.num_workers = num_workers

        self._init_dataset(train, test)

    def _init_dataset(self, train, test):
        train, val = train_test_split(train, test_size=0.2)
        print(
            f"Train : {train[self.label].value_counts()}\nval : {val[self.label].value_counts()}\nTest : {test[self.label].value_counts()}"
        )
        if self.y_type == "bt" and len(self.label) == 1:

            class_weights = dict(
                enumerate(
                    class_weight.compute_class_weight(
                        "balanced",
                        classes=np.arange(self.num_classes),
                        y=train[self.label[0]],
                    )
                )
            )
            self.class_weights = class_weights

        self.train = TableDataset(
            train, self.features, self.label, self.num_classes, self.y_type
        )
        self.validation = TableDataset(
            val, self.features, self.label, self.num_classes, self.y_type
        )
        self.test = TableDataset(
            test, self.features, self.label, self.num_classes, self.y_type
        )

    def train_dataloader(self):

        if self.y_type == "bt":
            train_class_weights = [
                self.class_weights[torch.argmax(i).item()] for i in self.train.y
            ]
            sampler = WeightedRandomSampler(
                train_class_weights, len(train_class_weights), replacement=True
            )
        else:
            sampler = RandomSampler(self.train)

        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            sampler=sampler,
            drop_last=True,
            persistent_workers=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation,
            batch_size=self.batch_size,
            persistent_workers=True,
            num_workers=self.num_workers,
            sampler=SequentialSampler(self.validation),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            persistent_workers=True,
            num_workers=self.num_workers,
            sampler=SequentialSampler(self.test),
        )



import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    auc,
)



def generate_multipletests_result(df, pvalue_col="pvalue", alpha=0.05, method="fdr_bh"):
    df = df.copy()
    pvalue_series = df[pvalue_col]
    reject, pvals_corrected, _, _ = multipletests(
        pvalue_series, alpha=alpha, method="fdr_bh"
    )
    df["pval_corrected"] = pvals_corrected
    df["reject"] = reject
    return df


def find_best_cutoff(fpr, tpr, thresholds):
    diff = tpr - fpr
    Youden_index = np.argmax(diff)
    optimal_threshold = thresholds[Youden_index]
    optimal_FPR, optimal_TPR = fpr[Youden_index], tpr[Youden_index]
    return optimal_threshold, optimal_FPR, optimal_TPR


def cal_binary_metrics(y, y_pred):
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    AUC = roc_auc_score(y, y_pred)
    # by best youden

    optim_threshold, optim_fpr, optim_tpr = find_best_cutoff(fpr, tpr, thresholds)
    y_pred_binary = (y_pred > optim_threshold).astype(int)
    ACC = accuracy_score(y, y_pred_binary)
    macro_f1 = f1_score(y, y_pred_binary, average="macro")
    sensitivity = optim_tpr
    specificity = 1 - optim_fpr
    precision, recall, _ = precision_recall_curve(y, y_pred)
    APR = auc(recall, precision)

    return {
        "AUC": AUC,
        "ACC": ACC,
        "Macro_F1": macro_f1,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "APR": APR,
    }




class LinearResBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearResBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.batch_norm = nn.LayerNorm(output_size)

        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")  # <6>
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)  # <7>
        torch.nn.init.zeros_(self.batch_norm.bias)

    def forward(self, x):
        out = self.fc1(x)

        out = self.batch_norm(out)
        out = torch.relu(out)
        return out + x

class FullyConnectedNet(pl.LightningModule):
    def __init__(
        self,
        hidden_size,
        features,
        output_size,
        num_resblocks=3,
        lr=1e-3,
        weight_decay=1e-2,
        weight=[1, 1],
        **kwargs,
    ):
        super(FullyConnectedNet, self).__init__()
        input_size = len(features)
        self.features = features
        self.norm = nn.LayerNorm(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.resblocks = nn.Sequential(
            *[LinearResBlock(hidden_size, hidden_size) for _ in range(num_resblocks)]
        )
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.lr = lr
        self.weight_decay = weight_decay

        self.mertic = {
            "train_auc": torchmetrics.AUROC(num_classes=2, task="multiclass"),
            "val_auc": torchmetrics.AUROC(num_classes=2, task="multiclass"),
        }
        self.history = defaultdict(dict)
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weight).float())

    def forward(self, x):
        x = self.norm(x)
        out = torch.relu(self.fc1(x))
        out = self.resblocks(out)
        out = self.fc2(out)
        return out

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        outputs = self.forward(x)
        loss = self.loss_fn(outputs, y.squeeze(-1).float())

        self.mertic["train_auc"].update(
            torch.softmax(outputs, dim=-1), torch.argmax(y, dim=1)
        )

        self.log("ptl/train_loss", loss, on_epoch=True, prog_bar=True, on_step=False)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        outputs = self.forward(x)
        loss = self.loss_fn(outputs, y.squeeze(-1).float())

        self.mertic["val_auc"].update(
            torch.softmax(outputs, dim=-1), torch.argmax(y, dim=1)
        )

        self.log("ptl/val_loss", loss, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self):

        auc = self.mertic["train_auc"].compute()
        self.log("ptl/train_auc", auc, prog_bar=True)

    def on_validation_epoch_end(self):
        auc = self.mertic["val_auc"].compute()
        self.log("ptl/val_auc", auc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def predict_df(self, df, batch_size=256):

        for feature in self.features:
            assert feature in df.columns
        print(f"input df have NA: {df['features'].isna().sum(axis=1).sum()}")
        df = df.copy().dropna(subset=self.features)

        predict_dataloader = DataLoader(
            torch.tensor(df[self.features].values).float(),
            batch_size=batch_size,
            persistent_workers=True,
            num_workers=4,
        )

        self.eval()
        pred = []
        with torch.no_grad():
            for x in predict_dataloader:
                y_hat = self.forward(x).cpu().detach()
                y_hat = torch.softmax(y_hat, dim=-1)[:, 1]

                pred.append(y_hat)
        pred = torch.cat(pred).numpy()
        df["pred"] = pred
        return df
    

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

    # The maximum training epochs
    num_epochs = 5

    # Number of sampls from parameter space
    num_samples = 10

    # tmp 
    proteomics = test_imputed.columns[test_imputed.columns.tolist().index("C3") :].tolist()

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
        dataset = DatasetModule(
            train=train_imputed,
            test=test_imputed,
            features=proteomics,
            label=["incident_cad"],
            num_classes=cpu_nums,
            batch_size=config["batch_size"],
            num_workers = cpu_nums,
        )
        # model = FullyConnectedNet(
        #     input_size=len(proteomics),
        #     hidden_size=config["hidden_size"],
        #     output_size=2,
        #     lr=config["lr"],
        #     weight_decay=config["weight_decay"],
        #     weight=config["weight"],
        #     num_resblocks=config["num_resblocks"],
        # )
        model = FullyConnectedNet(**config)
        trainer = Trainer(
            devices="auto",
            strategy=RayDDPStrategy(),
            callbacks=[RayTrainReportCallback()],
            plugins=[RayLightningEnvironment()],
            enable_progress_bar=True,
        )
        trainer = prepare_trainer(trainer)
        trainer.fit(model, dataset)


    search_space = {
        "features": proteomics,
        "output_size": 2,
        "hidden_size": tune.choice([32, 64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-4, 1e-1),
        "weight": tune.choice([[0.1, 1], [0.1, 10], [0.1, 100]]),
        "batch_size": tune.choice([64, 256]),
        "num_resblocks": tune.choice([2, 3, 4]),
    }



    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)


    scaling_config = ScalingConfig(
        num_workers=parallel_trile, use_gpu=True, resources_per_worker={"CPU": cpu_nums, "GPU": gpu_nums}
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
    best_model = FullyConnectedNet(**best_params["train_loop_config"])
    best_model.load_state_dict(best_model_state["state_dict"])