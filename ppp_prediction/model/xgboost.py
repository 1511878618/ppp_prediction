from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from ppp_prediction.plot import save_fig
from .basic import BaseModel
import pandas as pd
import xgboost as xgb
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import train_test_split
from ray import tune
from ray.tune.integration.xgboost import TuneReportCheckpointCallback


def get_best_model_checkpoint(results):
    best_result = results.get_best_result()

    # `TuneReportCheckpointCallback` provides a helper method to retrieve the
    # model from a checkpoint.
    best_bst = TuneReportCheckpointCallback.get_model(best_result.checkpoint)
    return best_bst


def tune_xgboost(
    train,
    X_var,
    label,
    val=None,
    smoke_test=False,
    max_iter=100,
    config=None,
    # device="cpu",
    # n_cpus=4,
    # n_gpus=0.5,
):
    search_space = {
        # You can mix constants with search space objects.
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error", "auc"],
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
    if val is None:
        train, val = train_test_split(train, test_size=0.2)

    def train_xgboost(config: dict):
        # Train the classifier, using the Tune callback
        train_set = xgb.DMatrix(train[X_var], label=train[label])
        val_set = xgb.DMatrix(val[X_var], label=val[label])
        xgb.train(
            config,
            train_set,
            evals=[(val_set, "eval")],
            verbose_eval=False,
            # `TuneReportCheckpointCallback` defines the checkpointing frequency and format.
            callbacks=[TuneReportCheckpointCallback(frequency=1)],
        )

    # if device == "cpu":
    #     n_gpus = 0

    tuner = tune.Tuner(
        # (
        #     train_xgboost
        #     if device == "cpu"
        #     else tune.with_resources(train_xgboost, num_cpus=4, num_gpus=0.5)
        # ),
        # tune.with_resources(train_xgboost, resources={"cpu": n_cpus, "gpu": n_gpus}),
        train_xgboost,
        tune_config=tune.TuneConfig(
            metric="eval-auc",
            mode="max",
            scheduler=scheduler,
            num_samples=1 if smoke_test else max_iter,
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    return results


class XGBoostModel(BaseModel):
    def __init__(
        self,
        mmconfig=None,
        dataconfig=None,
        tgtconfig=None,
        phenoconfig=None,
        testdataconfig=None,
    ):
        super(XGBoostModel, self).__init__(
            mmconfig=mmconfig,
            dataconfig=dataconfig,
            tgtconfig=tgtconfig,
            phenoconfig=phenoconfig,
            testdataconfig=testdataconfig,
        )

        self.modelname = "xgboost"

    def fit_or_tune(
        self,
        train,
        X_var,
        label,
        test=None,
        smoke_test=False,
        n_iter=2000,
        param_grid=None,
        **kwargs,
    ):
        result = tune_xgboost(
            train=train,
            X_var=X_var,
            label=label,
            max_iter=n_iter,
            val=test,
            smoke_test=smoke_test,
            config=param_grid,
        )
        best_model = get_best_model_checkpoint(result)

        if hasattr(best_model, "predict_proba"):
            train_pred = best_model.predict_proba(train[X_var])[:, 1]
            test_pred = best_model.predict_proba(test[X_var])[:, 1]
        else:
            train_pred = best_model.predict(train[X_var])
            test_pred = best_model.predict(test[X_var])

        train_pred_df = train[["eid", label]].copy()
        train_pred_df[f"pred_{label}"] = train_pred

        test_pred_df = test[["eid", label]].copy()
        test_pred_df[f"pred_{label}"] = test_pred

        return best_model, train_pred_df, test_pred_df, result

    def show_figs(self, model, train, test, X_var, label, outputFolder, **kwargs):
        import matplotlib.pyplot as plt

        figOutput = outputFolder / "figs"
        figOutput.mkdir(parents=True, exist_ok=True)
        # feature importance
        fig, ax = plt.subplots(figsize=(5, 10))
        xgb.plot_importance(model, ax=ax)
        save_fig(fig, figOutput / "feature_importance")
        data = pd.concat([train, test])

        # shap summary plot
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data[X_var])
        fig, ax = plt.subplots(figsize=(5, 10))
        shap.summary_plot(shap_values, data[X_var], max_display=20, show=False)
        save_fig(path=figOutput / "shap_summary_plot")

        # global importance
        shap.summary_plot(
            shap_values, data[X_var], plot_type="bar", max_display=20, show=False
        )
        save_fig(path=figOutput / "shap_summary_plot_bar")

        # shap dependence plot
        import numpy as np

        top_average_important_features_df = pd.DataFrame(
            {"feature": X_var, "shap": np.abs(shap_values).mean(0)}
        ).sort_values("shap", ascending=False)
        k = 10
        top_k_features = top_average_important_features_df["feature"].tolist()[:k]
        top_k_save_dir = figOutput / "top_features_dependence_plot"
        top_k_save_dir.mkdir(parents=True, exist_ok=True)
        for feature in top_k_features:
            shap.dependence_plot(
                feature,
                shap_values,
                data[X_var],
                display_features=data[X_var],
                show=False,
            )
            save_fig(path=top_k_save_dir / feature)


class XGBoostModel_V1(BaseModel):
    def __init__(
        self,
        mmconfig=None,
        dataconfig=None,
        tgtconfig=None,
        phenoconfig=None,
        testdataconfig=None,
    ):
        super(XGBoostModel_V1, self).__init__(
            mmconfig=mmconfig,
            dataconfig=dataconfig,
            tgtconfig=tgtconfig,
            phenoconfig=phenoconfig,
            testdataconfig=testdataconfig,
        )

        self.modelname = "xgboost"

    def fit_or_tune(
        self,
        train,
        test,
        X_var,
        label,
        param=None,
        param_grid=None,
        device="cuda",
        n_threads=4,
        n_iter=2000,
        **kwargs,
    ):
        """
        for binary classification
        """
        print(f"Using device: {device} with {n_threads} threads")
        param_default = {
            "eta": 0.3,
            # "objective": "binary:logistic",
            "objective": "reg:logistic",
            "nthread": n_threads,
            "eval_metric": "auc",
            "tree_method": "hist",
            "booster": "gbtree",
            "normalize_type": "tree",
            "class_weight": "balanced",
            "enable_categorical": True,
            "device": device,
            "random_state": 0,
            "learning_rate": 0.1,
            "scale_pos_weight": 1,
        }
        if param is not None:
            # param = {**param_default, **param}
            param.update(param_default)
        else:
            param = param_default

        param_grid_default = {
            "reg_alpha": [0, 0.1, 0.2, 0.5, 1, 4, 10],
            "reg_lambda": [0, 0.1, 0.2, 0.5, 1, 4, 10],
            "gamma": [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 10],
            "max_depth": [3, 5, 10, 30, 50, 100],
        }
        if param_grid is not None:
            param_grid = {**param_grid_default, **param_grid}
        else:
            param_grid = param_grid_default

        clf = XGBClassifier(**param)
        # clf = XGBClassifier(**param)
        grid_search = RandomizedSearchCV(
            clf,
            param_grid,
            cv=5,
            n_jobs=4,
            verbose=10,
            scoring="roc_auc",
            n_iter=n_iter,
            refit=True,
        )

        grid_search.fit(
            train[X_var],
            train[label],
        )
        print(grid_search.best_params_)

        best_model = grid_search.best_estimator_
        best_model.set_params(device="cpu")

        train_pred = best_model.predict_proba(train[X_var])[:, 1]
        test_pred = best_model.predict_proba(test[X_var])[:, 1]

        train_pred_df = train[["eid", label]].copy()
        train_pred_df[f"pred_{label}"] = train_pred

        test_pred_df = test[["eid", label]].copy()
        test_pred_df[f"pred_{label}"] = test_pred

        return best_model, train_pred_df, test_pred_df, grid_search

    def show_figs(self, model, train, test, X_var, label, outputFolder, **kwargs):
        import matplotlib.pyplot as plt

        figOutput = outputFolder / "figs"
        figOutput.mkdir(parents=True, exist_ok=True)
        # feature importance
        fig, ax = plt.subplots(figsize=(5, 10))
        xgb.plot_importance(model, ax=ax)
        save_fig(fig, figOutput / "feature_importance")
        data = pd.concat([train, test])

        # shap summary plot
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data[X_var])
        fig, ax = plt.subplots(figsize=(5, 10))
        shap.summary_plot(shap_values, data[X_var], max_display=20, show=False)
        save_fig(path=figOutput / "shap_summary_plot")

        # global importance
        shap.summary_plot(
            shap_values, data[X_var], plot_type="bar", max_display=20, show=False
        )
        save_fig(path=figOutput / "shap_summary_plot_bar")

        # shap dependence plot
        import numpy as np

        top_average_important_features_df = pd.DataFrame(
            {"feature": X_var, "shap": np.abs(shap_values).mean(0)}
        ).sort_values("shap", ascending=False)
        k = 10
        top_k_features = top_average_important_features_df["feature"].tolist()[:k]
        top_k_save_dir = figOutput / "top_features_dependence_plot"
        top_k_save_dir.mkdir(parents=True, exist_ok=True)
        for feature in top_k_features:
            shap.dependence_plot(
                feature,
                shap_values,
                data[X_var],
                display_features=data[X_var],
                show=False,
            )
            save_fig(path=top_k_save_dir / feature)
