import pandas as pd
from ppp_prediction.metrics import cal_binary_metrics
import numpy as np


# current
def run_glmnet(
    train,
    xvar,
    label,
    time=None,
    test=None,
    covariate=None,
    cv=5,
    alpha=1,
    lambda_=None,
    trace_it=1,
    family="gaussian",
    weights=None,
    type_measure="deviance",
    coef_choice="lambda.min",
    standardize=True,
    intercept=False,
    save_path=None,
):
    import rpy2.robjects as robjects

    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    import pandas as pd

    glmnet_script_path = __file__.replace("glmnet.py", "glmnet.R")
    robjects.r.source(glmnet_script_path)
    glmnet_func = robjects.r["glmnet_lasso"]
    pandas2ri.activate()
    # format conversion
    if covariate is not None and isinstance(covariate, list):
        covariate = robjects.StrVector(covariate)
    if xvar is not None and isinstance(xvar, list):
        xvar = robjects.StrVector(xvar)
    if weights is not None:
        if isinstance(weights, str):
            if weights == "balanced":
                from sklearn.utils import class_weight

                weights = class_weight.compute_sample_weight(
                    class_weight="balanced", y=train[label]
                )

        if isinstance(weights, pd.Series):
            weights = robjects.FloatVector(weights.values)
        elif isinstance(weights, np.ndarray):
            weights = robjects.FloatVector(weights)
        else:
            weights = robjects.FloatVector(weights)

    with localconverter(robjects.default_converter + pandas2ri.converter):
        result = glmnet_func(
            train,
            xvar,
            label,
            time if time is not None else robjects.r("NULL"),
            test if test is not None else robjects.r("NULL"),
            covariate if covariate is not None else robjects.r("NULL"),
            cv,
            alpha,
            lambda_ if lambda_ is not None else robjects.r("NULL"),
            trace_it,
            family,
            type_measure,
            coef_choice,
            standardize,
            intercept,
            weights=robjects.r("NULL") if weights is None else weights,
            save_path=save_path if save_path is not None else robjects.r("NULL"),
        )
        # result = [list(i) for i in result]
        train_mean = result["train_mean"]
        train_std = result["train_std"]
        train = result["train"]
        if train is not None:
            train = train.rename(columns={"pred": f"{label}_pred"})
            # train_metrics =
            try:
                to_cal_train = train[[label, f"{label}_pred"]].dropna()
                train_metrics = cal_binary_metrics(
                    to_cal_train[label], to_cal_train[f"{label}_pred"], ci=True
                )
            except:
                train_metrics = None
        if test is not None:
            # test.rename(columns={"pred": f"{label}_pred"})
            test = result["test"].rename(columns={"pred": f"{label}_pred"})
            try:
                to_cal_test = test[[label, f"{label}_pred"]].dropna()
                test_metrics = cal_binary_metrics(
                    to_cal_test[label], to_cal_test[f"{label}_pred"], ci=True
                )
            except:
                test_metrics = None
        else:
            test = None
        coef = result["coef"]
        # rm intercept
        keeped = coef.index.str.contains(".*Intercept.*")
        coef = coef[~keeped]
    model = glmnet_linear(
        coef,
        feature_names=xvar,
        mean=train_mean,
        std=train_std,
    )
    # train_metrics = cal_binary_metrics(train["y"], train["y_hat"])

    return (
        model,
        train_metrics,
        test_metrics,
        train,
        test,
    )


class glmnet_linear:
    def __init__(self, coef, intercept=None, feature_names=None, mean=None, std=None):
        self.coef = coef
        self.intercept = intercept if intercept is not None else 0
        self.feature_names = feature_names
        self.mean = mean
        self.std = std

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            if self.feature_names is not None:
                X = X[self.feature_names]
            else:
                X = X.values
        elif isinstance(X, pd.Series):
            X = X.values
        if self.mean is not None and self.std is not None:
            X = (X - self.mean) / self.std

        return X @ self.coef + self.intercept
