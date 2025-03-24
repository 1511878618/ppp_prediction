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