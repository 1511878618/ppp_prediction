from ppp_prediction.model import fit_best_model, fit_ensemble_model_simple
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
from ppp_prediction.metrics.common import cal_binary_metrics, cal_qt_metrics
from sklearn.model_selection import train_test_split
from ppp_prediction.model.glmnet import run_glmnet

# def run_glmnet_v2(
        

def get_predict_v2_from_df(
    model,
    data,
    x_var,
):
    """
    Use this to get prediction from model and data
    merge by idx
    """
    no_na_data = data[x_var].dropna().copy()

    if isinstance(model, xgb.core.Booster):
        no_na_data_DM = xgb.DMatrix(no_na_data)

        if hasattr(model, "predict_proba"):
            no_na_data["pred"] = model.predict_proba(no_na_data_DM)[:, 1]
        else:
            no_na_data["pred"] = model.predict(no_na_data_DM)
    else:

        if hasattr(model, "predict_proba"):
            no_na_data["pred"] = model.predict_proba(no_na_data)[:, 1]
        else:
            no_na_data["pred"] = model.predict(no_na_data)

    return (
        data[[]]
        .merge(no_na_data[["pred"]], left_index=True, right_index=True, how="left")
        .values.flatten()
    )


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







