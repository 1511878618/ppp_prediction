from ppp_prediction.model import fit_best_model, fit_ensemble_model_simple


def fit_best_model_v2(
    train,
    test,
    xvar,
    label,
    method_list="Lasso",
    cv=10,
    verbose=1,
    save_dir=None,
    engine: str = "cuml",
    y_type="bt",
):

    return fit_best_model(
        train,
        test,
        xvar,
        label,
        method_list,
        cv,
        verbose,
        save_dir,
        y_type,
        engine=engine,
    )


def fit_ensemble_model_simple_v2(
    train, test, xvar, label, engine="cuml", method="Linear", need_scale=False
):
    return fit_ensemble_model_simple(
        train, test, xvar, label, engine, method, need_scale
    )
