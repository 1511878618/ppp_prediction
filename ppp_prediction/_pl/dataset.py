
import torchvision
from pytorch_lightning import seed_everything

seed_everything(42)

from pytorch_lightning import LightningDataModule
import torch
from torch.nn import functional as F
import torch

import pandas as pd
import numpy as np
torchvision.disable_beta_transforms_warning()

import warnings

import torch
from pytorch_lightning import seed_everything




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

import numpy as np






import os

import torch



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


class TableDatasetModule(LightningDataModule):
    def __init__(
        self,
        train,
        test,
        val = None , 
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

        self._init_dataset(train, test, val = val)

    def _init_dataset(self, train, test, val=None):
        if val is None:
            train, val = train_test_split(train, test_size=0.2)
            print(f"split 0.2 of train to val")
        else:
            assert isinstance(val, pd.DataFrame)
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
