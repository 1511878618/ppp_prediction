from ppp_prediction.model import fit_best_model, fit_ensemble_model_simple
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
from ppp_prediction.metrics.common import cal_binary_metrics, cal_qt_metrics
from sklearn.model_selection import train_test_split
from ppp_prediction.model.glmnet import run_glmnet
import torch
from tqdm import tqdm
import numpy as np

import math

import time
from typing import Any, Optional

import optuna
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Module, MSELoss
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import AUROC, Accuracy, MeanSquaredError
from tqdm import tqdm

from torch_frame import stype
from torch_frame.data import DataLoader, Dataset

from torch_frame.nn.encoder import EmbeddingEncoder, LinearBucketEncoder
from torch_frame.nn.models import (
    MLP,
    ExcelFormer,
    FTTransformer,
    ResNet,
    TabNet,
    TabTransformer,
    Trompt,
)
from torch_frame.typing import TaskType
from copy import deepcopy
from sklearn.model_selection import train_test_split

# def run_glmnet_v2(


def get_predict_from_tb_dl_with_df(
    df,
    model,
    x_var,
    batch_size=2048,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    **kwargs,
):
    from torch_frame.data import Dataset
    from torch_frame.data import DataLoader

    # dataset
    df = df[x_var]
    col_to_stype = model.col_to_stype
    DataSetUsed = Dataset(
        df=df,
        col_to_stype=col_to_stype,
        # target_col=target_col,
    )
    DataSetUsed.materialize()
    DataSetUsedLoader = DataLoader(DataSetUsed, batch_size=batch_size, shuffle=False)

    #
    model.eval()
    with torch.no_grad():
        pred = []
        for batch in tqdm(DataSetUsedLoader):
            batch = batch.to(device)
            pred.append(model(batch).cpu().numpy())
        pred = np.concatenate(pred)
    return pred

import re

def get_predict_v2_from_df(model, data, x_var, **kwargs):
    """
    Use this to get prediction from model and data
    merge by idx
    """

    # check TabPFN in model._class__.__name__ ignoring the capital letter or not
    modelName = model.__class__.__name__
    if re.match(".*TabPFN.*", modelName) and modelName != "TunedTabPFNClassifier":
        if hasattr(model, "predict_proba"):
            pred = model.predict_proba(data[x_var])[:, 1]
        else:
            pred = model.predict(data[x_var])

    else:
        no_na_data = data[x_var].dropna().copy()
        if isinstance(model, xgb.core.Booster):
            no_na_data_DM = xgb.DMatrix(no_na_data)

            if hasattr(model, "predict_proba"):
                no_na_data["pred"] = model.predict_proba(no_na_data_DM)[:, 1]
            else:
                no_na_data["pred"] = model.predict(no_na_data_DM)
        elif hasattr(model, "col_to_stype"):
            no_na_data["pred"] = get_predict_from_tb_dl_with_df(
                df=no_na_data, model=model, x_var=x_var, **kwargs
            )

        else:
            if hasattr(model, "predict_proba"):
                no_na_data["pred"] = model.predict_proba(no_na_data)[:, 1]
            else:
                no_na_data["pred"] = model.predict(no_na_data)

        pred = (
            data[[]]
            .merge(no_na_data[["pred"]], left_index=True, right_index=True, how="left")
            .values.flatten()
        )
    return pred


def fit_tabular_dl(
    xvar,
    train,
    label,
    test=None,  # Test is validation
    cv=10,
    verbose=1,
    y_type="bt",
    need_scale=False,
    test_size=0.2,
    col_to_stype=None,  # cat_cols and qt_cols shouldb e stype.numerical
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    epochs=50,
    num_trials=10,
    # num_repeats=5,
    seed=42,
    model_type="TabNet",  # TabNet, TabTransformer, ExcelFormer, MLP, ResNet, Trompt, LightGBM, CatBoost, XGBoost
    **kwargs,
):
    """
    Fits a deep learning model for tabular data using various architectures.

    Args:
        xvar (list): List of feature column names
        train (pd.DataFrame): Training data
        label (str): Target column name
        test (pd.DataFrame, optional): Test/validation data. If None, splits train data
        cv (int, optional): Number of cross-validation folds. Defaults to 10
        verbose (int, optional): Verbosity level. Defaults to 1
        y_type (str, optional): Target type ('bt' for binary). Defaults to 'bt'
        need_scale (bool, optional): Whether to scale features. Defaults to False
        test_size (float, optional): Test split ratio if test=None. Defaults to 0.2
        col_to_stype (dict, optional): Column type mapping. Auto-inferred if None
        device (torch.device, optional): Device to run on. Defaults to GPU if available
        epochs (int, optional): Number of training epochs. Defaults to 50
        num_trials (int, optional): Number of hyperparameter tuning trials. Defaults to 10
        seed (int, optional): Random seed. Defaults to 42
        model_type (str, optional): Model architecture to use. Defaults to 'TabNet'
        **kwargs: Additional model-specific arguments

    Returns:
        tuple: (trained_model, results_dict)
            - trained_model: The fitted model
            - results_dict: Dictionary containing metrics and best configurations
    """
    if col_to_stype is None:
        col_to_stype = {}
        for col in xvar:
            if train[col].dtype in ["object", "category"]:
                if train[col].nunique() <= 2:
                    col_to_stype[col] = stype.categorical
                else:
                    col_to_stype[col] = stype.multicategorical
            else:
                col_to_stype[col] = stype.numerical

    train_df = train[xvar + [label]].copy().dropna().reset_index(drop=True)

    if test is not None:
        test_df = test[xvar + [label]].copy().dropna()
    else:
        print("No test data provided, will split the train data into train and test")
        train_df, test_df = train_test_split(
            train_df, test_size=test_size, random_state=42
        )

    train_dataset = Dataset(
        df=train_df,
        col_to_stype=col_to_stype,
        target_col=label,
    )
    train_dataset.materialize()
    val_dataset = Dataset(
        df=test_df,
        col_to_stype=col_to_stype,
        target_col=label,
    )
    val_dataset.materialize()

    # model_type = "MLP"
    # train_dataset = TrainDataSet
    # test_dataset = TestDataSet
    # val_dataset = ValDataSet
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_CONFIG_KEYS = ["batch_size", "gamma_rate", "base_lr"]
    # task_type = "binary_classification"  # binary_classification, multiclass_classification, regression
    # sacle = "small"  # small, medium, large
    # epochs = 10
    # num_trials = 3  # Number of Optuna-based hyper-parameter tuning.
    # num_repeats = 5  # Number of repeated training and eval on the best config
    # seed = 42

    torch.manual_seed(seed)

    train_tensor_frame = train_dataset.tensor_frame
    val_tensor_frame = val_dataset.tensor_frame
    # test_tensor_frame = test_dataset.tensor_frame

    if train_dataset.task_type == TaskType.BINARY_CLASSIFICATION:
        out_channels = 1
        loss_fun = BCEWithLogitsLoss()
        metric_computer = AUROC(task="binary").to(device)
        higher_is_better = True
    elif train_dataset.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        out_channels = train_dataset.num_classes
        loss_fun = CrossEntropyLoss()
        metric_computer = Accuracy(
            task="multiclass", num_classes=train_dataset.num_classes
        ).to(device)
        higher_is_better = True
    elif train_dataset.task_type == TaskType.REGRESSION:
        out_channels = 1
        loss_fun = MSELoss()
        metric_computer = MeanSquaredError(squared=False).to(device)
        higher_is_better = False

    # To be set for each model
    model_cls = None
    col_stats = None

    # Set up model specific search space
    if model_type == "TabNet":
        model_search_space = {
            "split_attn_channels": [64, 128, 256],
            "split_feat_channels": [64, 128, 256],
            "gamma": [1.0, 1.2, 1.5],
            "num_layers": [4, 6, 8],
        }
        train_search_space = {
            "batch_size": [
                2048,
                4096,
            ],  # Note if you have a small data, you may want to reduce it, also low gpu memory
            # "batch_size": [128, 256],
            "base_lr": [0.001, 0.01],
            "gamma_rate": [0.9, 0.95, 1.0],
        }
        model_cls = TabNet
        col_stats = train_dataset.col_stats
    elif model_type == "FTTransformer":
        model_search_space = {
            "channels": [64, 128, 256],
            "num_layers": [4, 6, 8],
        }
        train_search_space = {
            "batch_size": [256, 512],
            "base_lr": [0.0001, 0.001],
            "gamma_rate": [0.9, 0.95, 1.0],
        }
        model_cls = FTTransformer
        col_stats = train_dataset.col_stats
    elif model_type == "FTTransformerBucket":
        model_search_space = {
            "channels": [64, 128, 256],
            "num_layers": [4, 6, 8],
        }
        train_search_space = {
            "batch_size": [256, 512],
            "base_lr": [0.0001, 0.001],
            "gamma_rate": [0.9, 0.95, 1.0],
        }
        model_cls = FTTransformer

        col_stats = train_dataset.col_stats
    elif model_type == "ResNet":
        model_search_space = {
            "channels": [64, 128, 256],
            "num_layers": [4, 6, 8],
        }
        train_search_space = {
            "batch_size": [256, 512],
            "base_lr": [0.0001, 0.001],
            "gamma_rate": [0.9, 0.95, 1.0],
        }
        model_cls = ResNet
        col_stats = train_dataset.col_stats
    elif model_type == "MLP":
        model_search_space = {
            "channels": [64, 128, 256],
            "num_layers": [1, 2, 4],
        }
        train_search_space = {
            "batch_size": [256, 512],
            "base_lr": [0.0001, 0.001],
            "gamma_rate": [0.9, 0.95, 1.0],
        }
        model_cls = MLP
        col_stats = train_dataset.col_stats
    elif model_type == "TabTransformer":
        model_search_space = {
            "channels": [16, 32, 64, 128],
            "num_layers": [4, 6, 8],
            "num_heads": [4, 8],
            "encoder_pad_size": [2, 4],
            "attn_dropout": [0, 0.2],
            "ffn_dropout": [0, 0.2],
        }
        train_search_space = {
            "batch_size": [128, 256],
            "base_lr": [0.0001, 0.001],
            "gamma_rate": [0.9, 0.95, 1.0],
        }
        model_cls = TabTransformer
        col_stats = train_dataset.col_stats
    elif model_type == "Trompt":
        model_search_space = {
            "channels": [64, 128, 192],
            "num_layers": [4, 6, 8],
            "num_prompts": [64, 128, 192],
        }
        train_search_space = {
            "batch_size": [128, 256],
            "base_lr": [0.01, 0.001],
            "gamma_rate": [0.9, 0.95, 1.0],
        }
        if train_tensor_frame.num_cols > 20:
            # Reducing the model size to avoid GPU OOM
            model_search_space["channels"] = [64, 128]
            model_search_space["num_prompts"] = [64, 128]
        elif train_tensor_frame.num_cols > 50:
            model_search_space["channels"] = [64]
            model_search_space["num_prompts"] = [64]
        model_cls = Trompt
        col_stats = train_dataset.col_stats
    elif model_type == "ExcelFormer":
        from torch_frame.transforms import (
            CatToNumTransform,
            MutualInformationSort,
        )

        categorical_transform = CatToNumTransform()
        categorical_transform.fit(train_dataset.tensor_frame, train_dataset.col_stats)
        train_tensor_frame = categorical_transform(train_tensor_frame)
        # val_tensor_frame = categorical_transform(val_tensor_frame)
        # test_tensor_frame = categorical_transform(test_tensor_frame)
        col_stats = categorical_transform.transformed_stats

        mutual_info_sort = MutualInformationSort(task_type=train_dataset.task_type)
        mutual_info_sort.fit(train_tensor_frame, col_stats)
        train_tensor_frame = mutual_info_sort(train_tensor_frame)
        # val_tensor_frame = mutual_info_sort(val_tensor_frame)
        # test_tensor_frame = mutual_info_sort(test_tensor_frame)

        model_search_space = {
            "in_channels": [128, 256],
            "num_heads": [8, 16, 32],
            "num_layers": [4, 6, 8],
            "diam_dropout": [0, 0.2],
            "residual_dropout": [0, 0.2],
            "aium_dropout": [0, 0.2],
            "mixup": [None, "feature", "hidden"],
            "beta": [0.5],
            "num_cols": [train_tensor_frame.num_cols],
        }
        train_search_space = {
            "batch_size": [256, 512],
            "base_lr": [0.001],
            "gamma_rate": [0.9, 0.95, 1.0],
        }
        model_cls = ExcelFormer
    else:
        # TODO support more model
        raise NotImplementedError(f"model_type {model_type} not implemented")

    assert model_cls is not None
    assert col_stats is not None
    assert set(train_search_space.keys()) == set(TRAIN_CONFIG_KEYS)
    col_names_dict = train_tensor_frame.col_names_dict

    def train(
        model: Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> float:
        model.train()
        loss_accum = total_count = 0

        for tf in tqdm(loader, desc=f"Epoch: {epoch}"):
            tf = tf.to(device)
            y = tf.y
            if isinstance(model, ExcelFormer):
                # Train with FEAT-MIX or HIDDEN-MIX
                pred, y = model(tf, mixup_encoded=True)
            elif isinstance(model, Trompt):
                # Trompt uses the layer-wise loss
                pred = model(tf)
                num_layers = pred.size(1)
                # [batch_size * num_layers, num_classes]
                pred = pred.view(-1, out_channels)
                y = tf.y.repeat_interleave(num_layers)
            else:
                pred = model(tf)

            if pred.size(1) == 1:
                pred = pred.view(
                    -1,
                )
            if train_dataset.task_type == TaskType.BINARY_CLASSIFICATION:
                y = y.to(torch.float)
            loss = loss_fun(pred, y)
            optimizer.zero_grad()
            loss.backward()
            loss_accum += float(loss) * len(tf.y)
            total_count += len(tf.y)
            optimizer.step()
        return loss_accum / total_count

    @torch.no_grad()
    def test(
        model: Module,
        loader: DataLoader,
    ) -> float:
        model.eval()
        metric_computer.reset()
        for tf in loader:
            tf = tf.to(device)
            pred = model(tf)
            if isinstance(model, Trompt):
                pred = pred.mean(dim=1)
            if train_dataset.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                pred = pred.argmax(dim=-1)
            elif train_dataset.task_type == TaskType.REGRESSION:
                pred = pred.view(
                    -1,
                )
            metric_computer.update(pred, tf.y)
        return metric_computer.compute().item()

    def train_and_eval_with_cfg(
        model_cfg: dict[str, Any],
        train_cfg: dict[str, Any],
        trial: Optional[optuna.trial.Trial] = None,
    ) -> tuple[float, float]:
        # Use model_cfg to set up training procedure
        if model_type == "FTTransformerBucket":
            # Use LinearBucketEncoder instead
            stype_encoder_dict = {
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearBucketEncoder(),
            }
            model_cfg["stype_encoder_dict"] = stype_encoder_dict
        model = model_cls(
            **model_cfg,
            out_channels=out_channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
        ).to(device)
        model.reset_parameters()
        # Use train_cfg to set up training procedure
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["base_lr"])
        lr_scheduler = ExponentialLR(optimizer, gamma=train_cfg["gamma_rate"])
        train_loader = DataLoader(
            train_tensor_frame,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(val_tensor_frame, batch_size=train_cfg["batch_size"])
        # test_loader = DataLoader(test_tensor_frame, batch_size=train_cfg["batch_size"])

        if higher_is_better:
            best_val_metric = 0
        else:
            best_val_metric = math.inf

        for epoch in range(1, epochs + 1):
            train_loss = train(model, train_loader, optimizer, epoch)
            val_metric = test(model, val_loader)

            if higher_is_better:
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    # best_test_metric = test(model, test_loader)
            else:
                if val_metric < best_val_metric:
                    best_val_metric = val_metric
                    # best_test_metric = test(model, test_loader)
            lr_scheduler.step()
            print(f"Train Loss: {train_loss:.4f}, Val: {val_metric:.4f}")

            if trial is not None:
                trial.report(val_metric, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        # print(f"Best val: {best_val_metric:.4f}, Best test: {best_test_metric:.4f}")
        # return best_val_metric, best_test_metric
        print(f"Best val: {best_val_metric:.4f}")
        return best_val_metric, 0

    def objective(trial: optuna.trial.Trial) -> float:
        model_cfg = {}
        for name, search_list in model_search_space.items():
            model_cfg[name] = trial.suggest_categorical(name, search_list)
        train_cfg = {}
        for name, search_list in train_search_space.items():
            train_cfg[name] = trial.suggest_categorical(name, search_list)

        best_val_metric, _ = train_and_eval_with_cfg(
            model_cfg=model_cfg, train_cfg=train_cfg, trial=trial
        )
        return best_val_metric

    # Hyper-parameter optimization with Optuna
    print("Hyper-parameter search via Optuna")
    start_time = time.time()
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(),
        direction="maximize" if higher_is_better else "minimize",
    )
    study.optimize(objective, n_trials=num_trials)
    end_time = time.time()
    search_time = end_time - start_time
    print("Hyper-parameter search done. Found the best config.")
    params = study.best_params
    best_train_cfg = {}
    for train_cfg_key in TRAIN_CONFIG_KEYS:
        best_train_cfg[train_cfg_key] = params.pop(train_cfg_key)
    best_model_cfg = params

    print(
        # f"Repeat experiments {num_repeats} times with the best train "
        f"config {best_train_cfg} and model config {best_model_cfg}."
    )

    # retrain model
    if model_type == "FTTransformerBucket":
        # Use LinearBucketEncoder instead
        stype_encoder_dict = {
            stype.categorical: EmbeddingEncoder(),
            stype.numerical: LinearBucketEncoder(),
        }
        best_model_cfg["stype_encoder_dict"] = stype_encoder_dict

    model = model_cls(
        **best_model_cfg,
        out_channels=out_channels,
        col_stats=col_stats,
        col_names_dict=col_names_dict,
    ).to(device)
    model.reset_parameters()
    # Use train_cfg to set up training procedure
    optimizer = torch.optim.Adam(model.parameters(), lr=best_train_cfg["base_lr"])
    lr_scheduler = ExponentialLR(optimizer, gamma=best_train_cfg["gamma_rate"])
    train_loader = DataLoader(
        train_tensor_frame,
        batch_size=best_train_cfg["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_tensor_frame, batch_size=best_train_cfg["batch_size"])
    # test_loader = DataLoader(test_tensor_frame, batch_size=best_train_cfg["batch_size"])

    if higher_is_better:
        best_val_metric = 0
    else:
        best_val_metric = math.inf

    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, epoch)
        val_metric = test(model, val_loader)

        if higher_is_better:
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                # best_test_metric = test(model, test_loader)
        else:
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                # best_test_metric = test(model, test_loader)
        lr_scheduler.step()
        print(f"Train Loss: {train_loss:.4f}, Val: {val_metric:.4f}")

    # get_predict

    result_dict = {
        # 'args': __dict__,
        # "model": model,
        "best_val_metric": best_val_metric,
        # "best_test_metric": best_test_metric,
        "best_train_cfg": best_train_cfg,
        "best_model_cfg": best_model_cfg,
        "search_time": search_time,
    }
    col_to_stype_save = {k: v for k, v in deepcopy(col_to_stype).items() if k != label}
    model.col_to_stype = col_to_stype_save
    return model, result_dict


def fit_best_model_v2(
    train,
    xvar,
    label,
    test=None,
    method_list="Lasso",
    cv=10,
    verbose=1,
    save_dir=None,
    engine: str = "cuml",
    y_type="bt",
    **kwargs
):

    return fit_best_model(
        train_df=train,
        X_var=xvar,
        y_var=label,
        test_df=test,
        method_list=method_list,
        cv=cv,
        verbose=verbose,
        save_dir=save_dir,
        engine=engine,
        y_type=y_type,
    )


def fit_ensemble_model_simple_v2(
    train, xvar, label, test=None, engine="cuml", method="Linear", need_scale=False
):
    return fit_ensemble_model_simple(
        train_df=train,
        test_df=test,
        X_var=xvar,
        y_var=label,
        method=method,
        need_scale=need_scale,
        engine=engine,
    )


def fit_xgboost(
    train: pd.DataFrame,
    xvar: list,
    label: str,
    test: pd.DataFrame = None,
    task_type: str = "classification",
    test_size: float = 0.2,  # only work when test is None
    tuning=False,
    tune_config=None,
    params=None,
    # random_state=42
):
    """
    Example Code:

    from sklearn.datasets import load_breast_cancer
    import xgboost as xgb

    breast_df = load_breast_cancer(as_frame=True)["frame"]
    breast_df["target"] = breast_df["target"].astype(int)
    train_df, test_df = train_test_split(breast_df, test_size=0.2)

    xvar = breast_df.columns.tolist()[:-1]
    label = "target"

    model, train_metrics, test_metrics, train_df, test_df = fit_xgboost(
        train=train_df,
        xvar=xvar,
        label=label,
        # test=test_df,
        task_type="classification",
        tune=True,
        tune_config=dict(max_iter=10),
    )

    test_df["pred"] = model.predict(xgb.DMatrix(test_df[xvar]))
    test_metrics = cal_binary_metrics(test_df[label], test_df["pred"])
    print(f"External test metrics: {test_metrics}")

    """
    train_df = train[xvar + [label]].copy().dropna().reset_index(drop=True)
    if test is not None:
        test_df = test[xvar + [label]].copy().dropna()
    else:
        print("No test data provided, will split the train data into train and test")
        train_df, test_df = train_test_split(train_df, test_size=test_size)
    if not tuning:
        # run xgboost
        if task_type == "classification":
            scale_pos_weight = (len(train_df) - train_df[label].sum()) / train_df[
                label
            ].sum()
            print(f"scale_pos_weight: {scale_pos_weight}")
            # print(len(cls_weights))
            defalut_params = dict(
                n_estimators=500,
                max_depth=50,
                # objective="multi:softmax",
                objective="binary:logistic",
                random_state=1234,
                # num_class=2,
                # alpha=1,
                # eta=0.1,
                # gamma=1,
                eval_metric="auc",
                # sample_weight=cls_weights,
                # sample_weight=cls_weights,
                scale_pos_weight=scale_pos_weight,
                booster="gbtree",
                early_stopping_rounds=50,
                gpu_id=0,
            )
            if params is not None:
                defalut_params.update(params)
            model = xgb.XGBClassifier(**defalut_params)

        elif task_type == "regression":
            defalut_params = dict(
                n_estimators=500,
                max_depth=50,
                objective="reg:squarederror",
                random_state=1234,
                alpha=1,
                eta=0.1,
                gamma=0.1,
                eval_metric="rmse",
                booster="gbtree",
                early_stopping_rounds=50,
                gpu_id=0,
            )
            if params is not None:
                defalut_params.update(params)
            model = xgb.XGBRegressor(**defalut_params)

        else:
            raise ValueError("task_type should be classification or regression")
        train_DM = xgb.DMatrix(train_df[xvar], label=train_df[label])
        test_DM = xgb.DMatrix(test_df[xvar], label=test_df[label])

        booster = xgb.train(
            defalut_params,
            train_DM,
            evals=[(test_DM, "eval")],
            verbose_eval=False,
            # `TuneReportCheckpointCallback` defines the checkpointing frequency and format.
        )

        # turn the model to xgb model
        if task_type == "classification":
            model = xgb.XGBClassifier(**defalut_params)
            model.n_classes_ = 1
            model._Booster = booster
        elif task_type == "regression":
            model = xgb.XGBRegressor(**defalut_params)
            model._Booster = booster

        # model.fit(
        #     train_df[xvar],
        #     train_df[label],
        #     eval_set=[(test_df[xvar], test_df[label])],
        #     verbose=0,
        #     # early_stopping_rounds=50,
        # )
        results = None

    else:
        print("tune xgboost")
        # local import
        from ray.tune.integration.xgboost import TuneReportCheckpointCallback
        import ray
        from ray import tune
        from ray.tune.schedulers import ASHAScheduler

        def get_best_model_checkpoint(results):
            best_result = results.get_best_result()

            # `TuneReportCheckpointCallback` provides a helper method to retrieve the
            # model from a checkpoint.
            best_bst = TuneReportCheckpointCallback.get_model(best_result.checkpoint)
            return best_bst

        def tune_xgboost(
            train,
            xvar,
            label,
            test=None,
            smoke_test=False,
            task_type: str = "classification",
            max_iter=100,
            # device="cpu",
            n_cpus=10,
            n_gpus=0,
        ):

            ray.init(ignore_reinit_error=True, num_cpus=n_cpus, num_gpus=n_gpus)
            search_space = {
                # You can mix constants with search space objects.
                "objective": (
                    "binary:logistic"
                    if task_type == "classification"
                    else "reg:squarederror"
                ),
                "eval_metric": (
                    ["logloss", "error", "auc"]
                    if task_type == "classification"
                    else ["rmse"]
                ),
                # "max_depth": tune.randint(1, 30),
                "max_depth": tune.choice([3, 5, 10, 15, 30, 50, 100, 120]),
                "n_estimators": tune.randint(1, 1000),
                "min_child_weight": tune.choice([1, 2, 3]),
                "subsample": tune.uniform(0.5, 1.0),
                "eta": tune.loguniform(1e-4, 1e-1),
                "gamma": tune.uniform(0, 0.5),
                "reg_alpha": tune.loguniform(1e-4, 1e-1),
                "reg_lambda": tune.loguniform(1e-4, 1e-1),
                "class_weight": "balanced",
                "tree_method": "hist",
                "booster": "gbtree",
                # "device": device,
            }
            # This will enable aggressive early stopping of bad trials.
            scheduler = ASHAScheduler(
                max_t=50, grace_period=1, reduction_factor=2  # 10 training iterations
            )
            if test is None:
                train, test = train_test_split(train, test_size=0.2)

            def train_xgboost(config: dict, train, val):
                # Train the classifier, using the Tune callback
                train_set = xgb.DMatrix(train[xvar], label=train[label])
                val_set = xgb.DMatrix(val[xvar], label=val[label])
                xgb.train(
                    config,
                    train_set,
                    evals=[(val_set, "eval")],
                    verbose_eval=False,
                    # `TuneReportCheckpointCallback` defines the checkpointing frequency and format.
                    callbacks=[TuneReportCheckpointCallback(frequency=1)],
                )

            tuner = tune.Tuner(
                tune.with_parameters(train_xgboost, train=train, val=test),
                tune_config=tune.TuneConfig(
                    metric="eval-auc" if task_type == "classification" else "eval-rmse",
                    mode="max" if task_type == "classification" else "min",
                    scheduler=scheduler,
                    num_samples=1 if smoke_test else max_iter,
                ),
                param_space=search_space,
            )
            results = tuner.fit()
            ray.shutdown()

            return results

        tune_default_config = dict(max_iter=100, n_cpus=10, n_gpus=0)
        if tune_config is not None:
            tune_default_config.update(tune_config)
        results = tune_xgboost(
            train_df, xvar, label, test_df, **tune_default_config, task_type=task_type
        )

        booster = get_best_model_checkpoint(results)
        params = results.get_best_result().config  # get the best params

        # turn the model to xgb model
        if task_type == "classification":
            model = xgb.XGBClassifier(**params)
            model.n_classes_ = 1
            model._Booster = booster
        elif task_type == "regression":
            model = xgb.XGBRegressor(**params)
            model._Booster = booster

    if task_type == "classification":
        train_pred = model.predict_proba(train_df[xvar])[:, 1]
        test_pred = model.predict_proba(test_df[xvar])[:, 1]
    else:
        train_pred = model.predict(train_df[xvar])
        test_pred = model.predict(test_df[xvar])

    train_df[f"{label}_pred"] = train_pred
    test_df[f"{label}_pred"] = test_pred

    train_to_cal = train_df[[label, f"{label}_pred"]].dropna()
    test_to_cal = test_df[[label, f"{label}_pred"]].dropna()

    train_metrics = (
        cal_binary_metrics(train_to_cal[label], train_to_cal[f"{label}_pred"])
        if task_type == "classification"
        else cal_qt_metrics(train_to_cal[label], train_to_cal[f"{label}_pred"])
    )
    test_metrics = (
        cal_binary_metrics(test_to_cal[label], test_to_cal[f"{label}_pred"])
        if task_type == "classification"
        else cal_qt_metrics(test_to_cal[label], test_to_cal[f"{label}_pred"])
    )

    return model, train_metrics, test_metrics, train_df, test_df, results


def fit_lightgbm(
    train: pd.DataFrame,
    xvar: list,
    label: str,
    test: pd.DataFrame = None,
    task_type: str = "classification",
    test_size: float = 0.2,  # only work when test is None
    tuning=False,
    tune_config=None,
    params=None,
):
    """

    Example Code:

    from sklearn.datasets import load_breast_cancer

    breast_df = load_breast_cancer(as_frame=True)["frame"]
    breast_df["target"] = breast_df["target"].astype(int)
    train_df, test_df = train_test_split(breast_df, test_size=0.2)

    xvar = breast_df.columns.tolist()[:-1]
    label = "target"

    model, train_metrics, test_metrics, train_df, test_df = fit_lightgbm(
        train=train_df,
        xvar=xvar,
        label=label,
        # test=test_df,
        task_type="classification",
        tune=False,  # if True, will tune the model
        tune_config=dict(max_iter=10),
    )

    test_df["pred"] = model.predict(test_df[xvar])
    test_metrics = cal_binary_metrics(test_df[label], test_df["pred"])
    print(f"External test metrics: {test_metrics}")
    """

    # 数据处理
    train_df = train[xvar + [label]].copy().dropna().reset_index(drop=True)
    if test is not None:
        test_df = test[xvar + [label]].copy().dropna()
    else:
        print("No test data provided, will split the train data into train and test")
        train_df, test_df = train_test_split(train_df, test_size=test_size)
    # 检查是否需要进行超参数调优
    if not tuning:

        if task_type == "classification":
            # 处理类别不平衡的情况
            scale_pos_weight = (
                (len(train_df) - train_df[label].sum()) / train_df[label].sum()
                if train_df[label].sum() != 0
                else 1
            )

            print(f"scale_pos_weight: {scale_pos_weight}")
            # 构建 LightGBM 分类器
            default_params = dict(
                n_estimators=500,
                max_depth=50,
                metric=["binary_logloss", "auc"],
                objective="binary",
                random_state=1234,
                scale_pos_weight=scale_pos_weight,
                boosting_type="gbdt",
                verbose=-1,
            )
            if params is not None:
                default_params.update(params)
            model = lgb.LGBMClassifier(**default_params)
        elif task_type == "regression":
            # 构建 LightGBM 回归器
            default_params = dict(
                n_estimators=500,
                max_depth=50,
                metric=["rmse"],
                objective="regression",
                random_state=1234,
                boosting_type="gbdt",
                verbose=-1,
            )
            if params is not None:
                default_params.update(params)

            model = lgb.LGBMRegressor(**default_params)

        else:
            raise ValueError("task_type should be classification or regression")

        # 训练模型
        model.fit(
            train_df[xvar],
            train_df[label],
            eval_set=[(test_df[xvar], test_df[label])],
            callbacks=[lgb.early_stopping(stopping_rounds=5)],
        )

        # 进行预测
        if task_type == "classification":
            train_pred = model.predict_proba(train_df[xvar])[:, 1]
            test_pred = model.predict_proba(test_df[xvar])[:, 1]
        else:
            train_pred = model.predict(train_df[xvar])
            test_pred = model.predict(test_df[xvar])

        results = None

    else:
        # local import
        from ray.tune.integration.lightgbm import TuneReportCheckpointCallback
        import ray
        from ray import tune
        from ray.tune.schedulers import ASHAScheduler

        # local function
        def get_best_model_checkpoint(results):
            best_result = results.get_best_result()

            # `TuneReportCheckpointCallback` provides a helper method to retrieve the
            # model from a checkpoint.
            best_bst = TuneReportCheckpointCallback.get_model(best_result.checkpoint)
            return best_bst

        def tune_lightgbm(
            train_df,
            xvar,
            label,
            test_df=None,
            smoke_test=False,
            task_type: str = "classification",
            max_iter=100,
            n_cpus=10,
            n_gpus=0,
        ):

            # 初始化 Ray 资源
            ray.init(ignore_reinit_error=True, num_cpus=n_cpus, num_gpus=n_gpus)
            scale_pos_weight = (
                (len(train_df) - train_df[label].sum()) / train_df[label].sum()
                if train_df[label].sum() != 0
                else 1
            )
            # 定义搜索空间
            search_space = {
                "objective": (
                    "binary" if task_type == "classification" else "regression"
                ),
                "metric": (
                    ["auc", "binary_logloss", "binary_error"]
                    if task_type == "classification"
                    else ["rmse", "binary_error"]
                ),
                "boosting_type": tune.choice(["gbdt", "dart"]),
                "max_depth": tune.choice([3, 5, 10, 15, 30, 50, 100, 120]),
                "n_estimators": tune.randint(1, 1000),
                "min_child_weight": tune.choice([1, 2, 3]),
                "subsample": tune.uniform(0.5, 1.0),
                "learning_rate": tune.loguniform(1e-4, 1e-1),
                "reg_alpha": tune.loguniform(1e-4, 1e-1),
                "reg_lambda": tune.loguniform(1e-4, 1e-1),
                "num_leaves": tune.randint(10, 1000),
                "scale_pos_weight": scale_pos_weight,
                "device": "gpu" if n_gpus > 0 else "cpu",  # 自动选择设备
            }

            # 定义调度器，基于训练进度进行早停
            scheduler = ASHAScheduler(max_t=50, grace_period=1, reduction_factor=2)

            # 将训练数据集和验证数据集进行分割
            if test_df is None:
                train_df, test_df = train_test_split(train_df, test_size=0.2)

            # 定义 LightGBM 的训练函数
            def train_lightgbm(config: dict, train_df, val_df):
                # LightGBM 数据集
                train_set = lgb.Dataset(train_df[xvar], label=train_df[label])
                val_set = lgb.Dataset(
                    val_df[xvar], label=val_df[label], reference=train_set
                )

                # remap_metric_dict =
                remap_metric_dict = {
                    "binary_error": "eval-binary_error",
                }
                if task_type == "classification":
                    remap_metric_dict.update(
                        {
                            "auc": "eval-auc",
                            "binary_logloss": "eval-binary_logloss",
                        }
                    )
                else:
                    remap_metric_dict.update(
                        {
                            "rmse": "eval-rmse",
                        }
                    )
                # 使用 Tune 的回调进行报告和检查点保存
                lgb.train(
                    config,
                    train_set,
                    valid_sets=[val_set],
                    valid_names=["eval"],
                    callbacks=[
                        # TuneReportCheckpointCallback(frequency=1, filename="checkpoint")
                        lgb.early_stopping(stopping_rounds=5),
                        TuneReportCheckpointCallback(remap_metric_dict, frequency=1),
                    ],
                )

            # 使用 Ray Tune 进行超参数搜索
            tuner = tune.Tuner(
                tune.with_parameters(train_lightgbm, train_df=train_df, val_df=test_df),
                tune_config=tune.TuneConfig(
                    metric="auc" if task_type == "classification" else "rmse",
                    mode="max" if task_type == "classification" else "min",
                    # metric="binary_logloss",
                    # mode="min",
                    scheduler=scheduler,
                    num_samples=1 if smoke_test else max_iter,
                ),
                param_space=search_space,
            )

            # 运行调优
            results = tuner.fit()

            # 关闭 Ray 资源
            ray.shutdown()

            return results

        print("tune LightGBM")
        # 默认的调优配置
        tune_default_config = dict(max_iter=100, n_cpus=10, n_gpus=0)
        if tune_config is not None:
            tune_default_config.update(tune_config)
        # 使用调优逻辑
        results = tune_lightgbm(
            train_df, xvar, label, test_df, **tune_default_config, task_type=task_type
        )

        model = get_best_model_checkpoint(results)

        # 根据调优结果进行预测
        if hasattr(model, "predict_proba") and task_type == "classification":
            train_pred = model.predict_proba(train_df[xvar])[:, 1]
            test_pred = model.predict_proba(test_df[xvar])[:, 1]
        else:
            train_pred = model.predict(train_df[xvar])
            test_pred = model.predict(test_df[xvar])

    # 将预测结果添加到数据框中
    train_df[f"{label}_pred"] = train_pred
    test_df[f"{label}_pred"] = test_pred

    # 计算评价指标
    train_to_cal = train_df[[label, f"{label}_pred"]].dropna()
    test_to_cal = test_df[[label, f"{label}_pred"]].dropna()
    print(f"task_type: {task_type}", task_type == "classification")
    print(task_type == "classification")
    # 分类和回归的不同指标计算
    train_metrics = (
        cal_binary_metrics(train_to_cal[label], train_to_cal[f"{label}_pred"])
        if task_type == "classification"
        else cal_qt_metrics(train_to_cal[label], train_to_cal[f"{label}_pred"])
    )
    test_metrics = (
        cal_binary_metrics(test_to_cal[label], test_to_cal[f"{label}_pred"])
        if task_type == "classification"
        else cal_qt_metrics(test_to_cal[label], test_to_cal[f"{label}_pred"])
    )

    return model, train_metrics, test_metrics, train_df, test_df, results


import time
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def fit_SVM(
    train,
    xvar,
    label,
    test=None,
    cv=10,
    verbose=1,
    y_type="bt",
    need_scale=False,
    test_size=0.2,
    **kwargs,
):

    train_df = train[xvar + [label]].copy().dropna().reset_index(drop=True)
    if test is not None:
        test_df = test[xvar + [label]].copy().dropna()
    else:
        print("No test data provided, will split the train data into train and test")
        train_df, test_df = train_test_split(
            train_df, test_size=test_size, random_state=42
        )

    if y_type == "bt":
        model = SVC(kernel="linear", probability=True)
        train_df[label] = train_df[label].astype(int)
        test_df[label] = test_df[label].astype(int)

    elif y_type == "qt":
        model = SVR(kernel="linear")
    else:
        raise ValueError("y_type should be bt or qt, but", y_type)

    if need_scale:

        model = Pipeline([("scaler", StandardScaler()), ("model", model)])
    else:
        model = Pipeline([("model", model)])

    model.fit(train_df[xvar], train_df[label])

    # evaluate
    if hasattr(model, "predict_proba"):
        train_pred = model.predict_proba(train_df[xvar])[:, 1]
        test_pred = model.predict_proba(test_df[xvar])[:, 1]
    else:
        train_pred = model.predict(train_df[xvar])
        test_pred = model.predict(test_df[xvar])

    train_df[f"{label}_pred"] = train_pred
    test_df[f"{label}_pred"] = test_pred

    train_to_cal = train_df[[label, f"{label}_pred"]].dropna()
    test_to_cal = test_df[[label, f"{label}_pred"]].dropna()

    if y_type == "bt":
        train_metrics = cal_binary_metrics(
            train_to_cal[label], train_to_cal[f"{label}_pred"]
        )
        test_metrics = cal_binary_metrics(
            test_to_cal[label], test_to_cal[f"{label}_pred"]
        )
    elif y_type == "qt":
        train_metrics = cal_qt_metrics(
            train_to_cal[label], train_to_cal[f"{label}_pred"]
        )
        test_metrics = cal_qt_metrics(test_to_cal[label], test_to_cal[f"{label}_pred"])

    return model, train_metrics, test_metrics, train_df, test_df

from tabpfn import TabPFNClassifier, TabPFNRegressor


def fit_rfpfn(
    xvar,
    train,
    label,
    y_type="bt",
    downsample_strategy="balance",
    device="cuda",
    **kwargs,
):
    # device = kwargs.get("device", "cpu")

    train_df = train[xvar + [label]].copy().dropna().reset_index(drop=True)

    train_nums = len(train_df)
    N_features = len(xvar)
    if N_features > 150:
        raise ValueError("N_features should be less than 150, but", N_features)
    from tabpfn_extensions.rf_pfn import (
        RandomForestTabPFNClassifier,
        RandomForestTabPFNRegressor,
    )

    clf_base = (
        TabPFNClassifier(
            device="cuda:0" if device == "cuda" else "cpu",
        )
        if y_type == "bt"
        else TabPFNRegressor(
            device="cuda:0" if device == "cuda" else "cpu",
        )
    )

    RFTabPFNClass = (
        RandomForestTabPFNClassifier if y_type == "bt" else RandomForestTabPFNRegressor
    )
    model = RFTabPFNClass(
        tabpfn=clf_base,
        n_estimators=100,
        max_depth=5,  # Use shallow trees for faster training
    )

    if y_type == "bt":
        train_df[label] = train_df[label].astype(int)

        if train_nums > 10000:
            print(f"train_nums: {train_nums}, will downsample to 10000")
            if downsample_strategy == "balance":
                min_catagory_nums = train_df[label].value_counts().min()
                min_catagory_nums = min(
                    5000, min_catagory_nums
                )  # binary job, no more than 5000
                train_df = (
                    train_df.groupby(label)
                    .apply(lambda x: x.sample(min_catagory_nums, random_state=42))
                    .reset_index(drop=True)
                )
                print(f"label after downsample: {train_df[label].value_counts()}")
            elif downsample_strategy == "random":
                train_df = train_df.sample(10000, random_state=42)
            else:
                raise ValueError(
                    "downsample_strategy should be balance or random, but ",
                    downsample_strategy,
                )

    elif y_type == "qt":
        if train_nums > 10000:
            print(f"train_nums: {train_nums}, will downsample to 10000")
            train_df = train_df.sample(10000, random_state=42)

    else:
        raise ValueError("y_type should be bt or qt, but", y_type)

    model.fit(train_df[xvar], train_df[label])

    return (model,)


def fit_tabpfn(
    xvar,
    train,
    label,
    y_type="bt",
    downsample_strategy="balance",
    device="cuda",
    tune=False,
    **kwargs,
):
    # device = kwargs.get("device", "cpu")

    train_df = train[xvar + [label]].copy().dropna().reset_index(drop=True)

    train_nums = len(train_df)
    N_features = len(xvar)
    if N_features > 150:
        raise ValueError("N_features should be less than 150, but", N_features)
    if tune:
        from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (
            AutoTabPFNClassifier,
            AutoTabPFNRegressor,
        )
    if y_type == "bt":
        model = (
            TabPFNClassifier(
                device="cuda:0" if device == "cuda" else "cpu",
                ignore_pretraining_limits=True,
                memory_saving_mode=True,
            )
            if not tune
            else AutoTabPFNClassifier(
                max_time=40,
                device="cuda" if device == "cuda" else "cpu",
                ignore_pretraining_limits=True,
            )
        )

        train_df[label] = train_df[label].astype(int)

        if train_nums > 10000:
            print(f"train_nums: {train_nums}, will downsample to 10000")
            if downsample_strategy == "balance":
                min_catagory_nums = train_df[label].value_counts().min()
                min_catagory_nums = min(
                    5000, min_catagory_nums
                )  # binary job, no more than 5000
                train_df = (
                    train_df.groupby(label)
                    .apply(lambda x: x.sample(min_catagory_nums, random_state=42))
                    .reset_index(drop=True)
                )
                print(f"label after downsample: {train_df[label].value_counts()}")
            elif downsample_strategy == "random":
                train_df = train_df.sample(10000, random_state=42)
            else:
                raise ValueError(
                    "downsample_strategy should be balance or random, but ",
                    downsample_strategy,
                )

    elif y_type == "qt":
        model = (
            TabPFNRegressor(
                device="cuda:0" if device == "cuda" else "cpu",
                ignore_pretraining_limits=True,
            )
            if not tune
            else AutoTabPFNRegressor(
                max_time=40,
                device="cuda" if device == "cuda" else "cpu",
                ignore_pretraining_limits=True,
            )
        )

        if train_nums > 10000:
            print(f"train_nums: {train_nums}, will downsample to 10000")
            train_df = train_df.sample(10000, random_state=42)

    else:
        raise ValueError("y_type should be bt or qt, but", y_type)

    model.fit(train_df[xvar], train_df[label])

    return (model,)
