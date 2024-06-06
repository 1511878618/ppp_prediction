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

    glmnet_script_path = "glmnet.R"
    robjects.r.source(glmnet_script_path)
    glmnet_func = robjects.r["glmnet_lasso"]
    pandas2ri.activate()
    # format conversion
    if covariate is not None and isinstance(covariate, list):
        covariate = robjects.StrVector(covariate)
    if xvar is not None and isinstance(xvar, list):
        xvar = robjects.StrVector(xvar)

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
            save_path if save_path is not None else robjects.r("NULL"),
        )
        # result = [list(i) for i in result]
        train_mean = result["train_mean"]
        train_std = result["train_std"]
        train = result["train"]
        test = result["test"]
        coef = result["coef"]
    return {
        "train_mean": train_mean,
        "train_std": train_std,
        "train": train,
        "test": test,
        "coef": coef,
    }