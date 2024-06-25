from sklearn.model_selection import GridSearchCV
from ppp_prediction.plot import save_fig
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, make_scorer
import numpy as np
from .basic import BaseModel


class LinearModel(BaseModel):
    def __init__(
        self,
        mmconfig=None,
        dataconfig=None,
        tgtconfig=None,
        phenoconfig=None,
        testdataconfig=None,
    ):
        super(LinearModel, self).__init__(
            mmconfig=mmconfig,
            dataconfig=dataconfig,
            tgtconfig=tgtconfig,
            phenoconfig=phenoconfig,
            testdataconfig=testdataconfig,
        )
        self.modelname = None

    def run(self, **kwargs):
        modelname = kwargs.pop("modelname", "Logistic")
        self.modelname = modelname
        super().run(**kwargs)

    def fit_or_tune(
        self,
        train,
        test,
        X_var,
        label,
        param=None,
        modelname="Logistic",
        # param_grid=None,
        device="cuda",
        n_threads=4,
        cv=5,
        # n_iter=100,
        **kwargs,
    ):
        """
        for binary classification
        """
        if device == "cuda":
            from cuml.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
        else:
            from sklearn.linear_model import (
                LogisticRegression,
                Ridge,
                Lasso,
                ElasticNet,
            )

        if hasattr(self, "modelname"):
            modelname = self.modelname
        print(f"Runing {modelname},using device: {device} with {n_threads} threads")

        print(f"Using device: {device} with {n_threads} threads")

        models_params = {
            "Logistic": {
                "model": LogisticRegression(
                    solver="liblinear" if device == "cpu" else "qn",
                    random_state=42,
                    class_weight="balanced",
                ),
                "param_grid": {
                    "C": np.logspace(-4, 4, 10),  # C参数的范围，使用对数间隔
                    "penalty": ["l1"],  # 正则化类型
                },
            },
            "Lasso": {
                "model": Lasso(),
                "param_grid": {
                    "alpha": np.logspace(-6, 2, 10),
                },
            },
            "ElasticNet": {
                "model": ElasticNet(),
                "param_grid": {
                    "alpha": np.logspace(-4, 2, 5),
                    "l1_ratio": np.linspace(0, 1, 5),
                },
            },
            "Ridge": {
                "model": Ridge(),
                "param_grid": {
                    "alpha": np.logspace(-6, 2, 10),
                },
            },
        }
        if modelname not in models_params:
            raise ValueError(
                f"modelname: {modelname} not in {models_params.keys()}"
                + f"supported models: {models_params.keys()}"
            )
        if modelname == "Logistic":
            scorer = make_scorer(roc_auc_score, needs_proba=True)
        else:
            scorer = make_scorer(roc_auc_score)

        clf = models_params[modelname]["model"]
        clf = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        param_grid = models_params[modelname]["param_grid"]
        param_grid = {f"clf__{k}": v for k, v in param_grid.items()}
        grid_search = GridSearchCV(
            clf,
            param_grid,
            cv=cv,
            n_jobs=n_threads,
            verbose=10,
            # scoring="roc_auc",
            scoring=scorer,
            # n_iter=n_iter,
            refit=True,
        )
        # as type int
        train[label] = train[label].astype(int)
        test[label] = test[label].astype(int)

        grid_search.fit(
            train[X_var],
            train[label],
        )
        print(grid_search.best_params_)

        best_model = grid_search.best_estimator_

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

        return best_model, train_pred_df, test_pred_df, grid_search
