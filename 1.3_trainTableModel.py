#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Description:       :
@Date     :2023/11/16 19:52:09
@Author      :Tingfeng Xu
@version      :2.0
"""
# TODO:完善成模板
from timm.optim import create_optimizer_v2
from timm.scheduler import CosineLRScheduler
from timm.scheduler.scheduler import Scheduler
import datetime
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning import loggers as pl_loggers

from collections import defaultdict
from pytorch_lightning import Trainer, seed_everything
import argparse

seed_everything(42)

from pytorch_lightning import LightningModule, LightningDataModule
import torch
from pytorch_lightning import trainer, LightningModule
from torch.nn import functional as F
import torch
import torchmetrics
from torch import nn
import pandas as pd
import numpy as np
import argparse

import textwrap
import warnings
import torchmetrics
from torchmetrics.aggregation import MeanMetric

warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

from torchmetrics.aggregation import CatMetric
from collections import defaultdict
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
from pytorch_lightning import Callback
from pathlib import Path
import pandas as pd

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_recall_curve,
    auc,
)


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


class HistoryCallback(Callback):

    def __init__(self, rootPath, **kwargs) -> None:
        super().__init__()

        self.log_dir = Path(rootPath) / "history/history.csv"
        self.log_dir.parent.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_end(self, trainer, module):
        history = pd.DataFrame(module.history)
        history.to_csv(str(self.log_dir), index=False)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        # return super().on_test_epoch_end(trainer, pl_module)
        history = pd.DataFrame(pl_module.history)
        history.to_csv(str(self.log_dir), index=False)


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


class TableModel(LightningModule):
    def __init__(self, input_size, num_classes, lr=1e-1):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
        self.lr = lr
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=torch.Tensor([0.5, 0.5]))
        # self.loss_fn = FocalLoss(alpha=torch.Tensor([1, 0.1]), gamma=2)
        self.history = defaultdict(list)

        self._init_metrics()

    def _init_metrics(self):
        self.train_metrics = {
            f"train_loss": MeanMetric(),
            "train_y": CatMetric(),
            "train_pred": CatMetric(),
        }
        self.val_metrics = {
            f"val_loss": MeanMetric(),
            "val_y": CatMetric(),
            "val_pred": CatMetric(),
        }
        self.test_metrics = {
            f"test_loss": MeanMetric(),
            "test_y": CatMetric(),
            "test_pred": CatMetric(),
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        x, y = batch
        # x = self.norm(x)
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y.float())

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    def cal_metrics(self, stage="train"):
        current_metrics = getattr(self, f"{stage}_metrics")
        loss = current_metrics[f"{stage}_loss"].compute()
        pred = current_metrics[f"{stage}_pred"].compute()
        y = current_metrics[f"{stage}_y"].compute()

        current_metric_kw = cal_binary_metrics(
            y.cpu().detach().numpy(), pred.cpu().detach().numpy()
        )
        current_metric_kw[f"loss"] = loss

        for k, v in current_metric_kw.items():
            key_name = f"{stage}_{k}"

            self.log_dict({key_name: v}, prog_bar=True, on_epoch=True)
            # update to history
            self.history[k].append(v.item())
        self.history["stage"].append(stage)
        self.history["epoch"].append(self.current_epoch)

    def on_train_epoch_end(self):

        self.cal_metrics(stage="train")

        # pass

    def on_validation_epoch_end(self):
        self.cal_metrics(stage="val")

    def on_test_epoch_end(self):
        self.cal_metrics(stage="test")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # x = self.norm(x)
        y = y.squeeze(-1).long()
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y.float())
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(
            self.model,
            opt=self.hparams.opt,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams["loss"] == "BMCLoss":
            optimizer.add_param_group(
                {
                    "params": self.loss_fn.noise_sigma,
                    "lr": self.hparams["sigma_lr"],
                    "name": "noise_sigma",
                }
            )

        # scheduler, _ = create_scheduler(self.hparams, optimizer)
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=self.hparams.t_initial,
            lr_min=self.hparams.lr_min,
            warmup_lr_init=self.hparams.warmup_lr_init,
            warmup_t=self.hparams.warmup_t,
            k_decay=self.hparams.k_decay,
            # cycle_limit=(self.hparams.epochs // self.hparams.t_initial) + 1,
            cycle_limit=self.hparams.cycle_limit,
        )
        scheduler_dict = {
            "scheduler": scheduler,
            "interval": self.hparams.update_lr_by,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

    def lr_scheduler_step(self, scheduler: Scheduler, metric) -> None:
        current_fake_epoch = (
            self.global_step / self.trainer.num_training_batches
            if self.hparams.update_lr_by == "step"
            else self.current_epoch
        )
        scheduler.step(
            epoch=current_fake_epoch
        )  # timm's scheduler need the epoch value, update by epochs


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
    parser.add_argument(
        "-m", "--model", type=str, default="TableModel", help="Model name"
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default="output", help="Output dir"
    )
    parser.add_argument(
        "-e",
        "--earily_stop",
        action="store_true",
        help="Whether to use earily stop",
    )
    return parser


if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()
    # For Pretrain dict

    import json
    import pandas as pd

    train_imputed = pd.read_pickle("1_train_imputed.pkl")
    test_imputed = pd.read_pickle("1_test_imputed.pkl")
    combination_json = json.load(open("1_X_combination_dict.json"))
    proteins = combination_json["all_protein"]
    proteins

    test_imputed_dataset = TableDataset(test_imputed, proteins, ["incident_cad"])

    test_imputed_dataset[1]
    model = TableModel(input_size=len(proteins), num_classes=2, lr=1e-1)
    dataset = DatasetModule(
        train=train_imputed,
        test=test_imputed,
        features=proteins,
        label=["incident_cad"],
        num_classes=2,
        batch_size=64,
    )

    dirpath = args.output_dir
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    version = args.model + f"-{current_time}"  # save dir

    log_dir = f"{dirpath}/lightning_logs/{version}"

    historyCallback = HistoryCallback(rootPath=log_dir)
    csvlogger = CSVLogger(save_dir=dirpath, version=version)
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir,
        filename="{epoch}-{val_loss:.2f}-{train_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        # save one epoch one model
        # save_top_k=-1,
        # every_n_epochs=1,
        save_last=True,
    )
    logging_interval = "epoch"

    callbacks = [
        # StochasticWeightAveraging(
        #     swa_lrs=1e-2
        # ),  # https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging
        LearningRateMonitor(logging_interval=logging_interval),
        TQDMProgressBar(refresh_rate=15),
        checkpoint_callback,
        historyCallback,
    ]
    if args.earily_stop:
        callbacks.append(
            EarlyStopping(monitor="val_loss", patience=10, verbose=True, mode="min")
            if args.earily_stop
            else None
        )

    Trainer = trainer.Trainer(
        max_epochs=100,
        accelerator="auto",
        precision="bf16",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        logger=[
            csvlogger,
            pl_loggers.TensorBoardLogger(save_dir=dirpath, version=version),
        ],
        callbacks=callbacks,
    )
    Trainer.fit(model, dataset)
