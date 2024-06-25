from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from ppp_prediction.plot import save_fig
from .basic import BaseModel
import pandas as pd
import xgboost as xgb


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
        test,
        X_var,
        label,
        param=None,
        param_grid=None,
        device="cuda",
        n_threads=4,
        n_iter=25,
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
            "reg_alpha": [0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 4, 6, 8, 10],
            "reg_lambda": [0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 4, 6, 8, 10],
            "gamma": [0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 4, 6, 8, 10],
            "max_depth": [10, 20, 50, 100, 150],
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
