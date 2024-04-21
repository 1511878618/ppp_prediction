

import torch


from pytorch_lightning import seed_everything

seed_everything(42)



from torch import nn
import os

import pytorch_lightning as pl

import torch.optim as optim


class LinearResBlock(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super(LinearResBlock, self).__init__()
        if d_ff is None:
            d_ff = d_model * 2

        self.fc1 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # d_model => d_model (default is d_model)
        x = self.norm1(x)
        x = x + self.dropout1(self.fc1(x))
        x = self.norm2(x)
        x = x + self.dropout2(self.ff(x))
        return x


class LinearTransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff=None, num_classes=0, num_layers=3, dropout=0.1):
        super(LinearTransformerEncoder, self).__init__()
        self.layers = nn.Sequential(
            *[
                LinearResBlock(d_model, d_ff=d_ff, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.fc_norm = (
            nn.LayerNorm(d_model, eps=1e-6) if num_classes > 0 else nn.Identity()
        )
        self.head_drop = nn.Dropout(dropout)
        self.head = (
            nn.Linear(d_model, num_classes) if num_classes > 0 else nn.Identity()
        )

        self.num_classes = num_classes
        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, x):
        x = self.layers(x)

        x = self.fc_norm(x)
        x = self.head_drop(x)
        x = self.head(x)
        return x


class LinearActivationNormDropOut(nn.Module):
    def __init__(self, d_in, d_out, activation=nn.SiLU(), dropout=0.1):
        super(LinearActivationNormDropOut, self).__init__()
        self.fc = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class LinearFeatureExtractor(nn.Module):
    def __init__(self, d_model, d_ff=None, d_out=None, dropout=0.1):
        super(LinearFeatureExtractor, self).__init__()
        if d_ff is None:
            d_ff = d_model * 2
        if d_out is None:
            d_out = d_model

        self.extractor = nn.Sequential(
            LinearActivationNormDropOut(
                d_model, d_ff, activation=nn.SiLU(), dropout=dropout
            ),
            LinearActivationNormDropOut(
                d_ff, d_ff, activation=nn.SiLU(), dropout=dropout
            ),
            LinearActivationNormDropOut(
                d_ff, d_out, activation=nn.SiLU(), dropout=dropout
            ),
        )

    def forward(self, x):
        return self.extractor(x)


class LinearFeatureFusionBlock(nn.Module):
    def __init__(
        self,
        d_model_list,
        d_ff=128,  # median layer dim. recommanded not to higher than any of d_model passed
        d_out=128,
        dropout=0.1,
        fusion_method="add",
    ):
        """
        fusion_method: add
        """
        super(LinearFeatureFusionBlock, self).__init__()
        self.EachPartModuleList = nn.ModuleList(
            [
                LinearFeatureExtractor(
                    d_model, d_ff=d_model * 2, d_out=d_ff, dropout=dropout
                )
                for d_model in d_model_list
            ]
        )
        self.d_ff = d_ff
        self.fusionDecoder = LinearFeatureExtractor(
            d_ff, d_ff=d_ff * 2, d_out=d_out, dropout=dropout
        )

        self.d_model_list = d_model_list

        self.d_out = d_out
        self.dropout = dropout
        self.fusion_method = fusion_method

    def intermidiate_forward(self, *x_list):
        return [module(x) for module, x in zip(self.EachPartModuleList, x_list)]

    def forward(self, *x_list):
        x_list = self.intermidiate_forward(*x_list)

        if self.fusion_method == "add":
            x = torch.stack(x_list, dim=-1).sum(dim=-1)
        # elif self.fusion_method == "concat":
        #     x = torch.cat(x_list, dim=-1)
        # elif self.fusion_method == "minus":
        else:
            raise NotImplementedError("Not implemented")

        x = self.fusionDecoder(x)
        return x


class LinearTransformer(nn.Module):
    def __init__(
        self,
        # features_dict,
        # covariates_dict=None,
        features,
        covariates=None
        d_ff=128,
        num_classes=2,
        num_layers=3,
        dropout=0.1,
    ):
        """
        lt = LinearTransformer(
        features_dict={"proteomics": proteomics},
        covariates_dict={"risk_factors": risk_factors},
        d_ff=512,
        num_classes=2,
        num_layers=2,
        dropout=0.1,
    )
        
        """
        super(LinearTransformer, self).__init__()
        # self.features_dict = features_dict
        # self.covariates_dict = covariates_dict if covariates_dict else None
        # self.features_name = list(features_dict.keys())[0]
        # self.covariates_name = (
        #     list(covariates_dict.keys())[0] if covariates_dict else None
        # )
        # self.features = features_dict[self.features_name]
        # self.covariates = (
        #     covariates_dict[self.covariates_name] if covariates_dict else None
        # )
        self.features = features
        self.covariates = covariates if covariates_dict else None


        self.d_featurs = len(self.features)
        self.d_covariates = len(self.covariates) if covariates_dict else None
        self.d_ff = d_ff if d_ff else self.d_featurs

        self.encoder = LinearTransformerEncoder(
            self.d_featurs,
            d_ff=self.d_ff,
            num_classes=d_ff,
            num_layers=num_layers,
            dropout=dropout,
        )  # d_features => d_ff

        d_model_list = (
            [self.d_ff, self.d_covariates]
            if self.d_covariates is not None
            else [self.d_ff]
        )
        self.decoder = LinearFeatureFusionBlock(
            d_model_list=d_model_list, d_out=d_ff, dropout=dropout
        )

        self.fc_norm = (
            nn.LayerNorm(d_ff, eps=1e-6) if num_classes > 0 else nn.Identity()
        )
        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_ff, num_classes) if num_classes > 0 else nn.Identity()

    def run_encoder(self, x):
        return self.encoder(x)

    def forward(self, x, cov=None):

        x = self.run_encoder(x)

        if cov is not None:
            out = self.decoder(x, cov)
        else:
            out = self.decoder(x)

        out = self.fc_norm(out)
        out = self.head_drop(out)
        out = self.head(out)

        return out