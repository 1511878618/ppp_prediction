import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import torch
import pytorch_lightning as pl

import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import torchmetrics
from .model import LinearTransformer
from torch.utils.data import DataLoader

class LinearTransformerPL(pl.LightningModule):
    def __init__(
        self,
        # features_dict,
        # covariates_dict=None,
        features,
        covariates=None,
        d_ff=512,
        num_classes=2,
        num_layers=2,
        dropout=0.1,
        lr=1e-3,
        weight_decay=1e-2,
        weight=[1, 1],
        **kwargs,
    ):

        super(LinearTransformerPL, self).__init__()

        self.lr = lr
        self.weight_decay = weight_decay

        self.mertic = {
            "train_auc": torchmetrics.AUROC(num_classes=2, task="multiclass"),
            "val_auc": torchmetrics.AUROC(num_classes=2, task="multiclass"),
        }
        self.history = defaultdict(dict)
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weight).float())
        self.model = LinearTransformer(
            # features_dict=features_dict,
            # covariates_dict=covariates_dict,
            features=features,
            covariates=covariates,
            d_ff=d_ff,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.features =  features

    def forward(self, x):

        return self.model(*x) if isinstance(x, (list, tuple)) else self.model(x)

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
        print(f"input df have NA: {df[self.features].isna().sum(axis=1).sum()}")
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