from cuml import LogisticRegression, Lasso, Ridge, ElasticNet

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
import numpy as np
import pandas as pd 

def fit_best_model(train_df, test_df, X_var, y_var, method_list=None, cv=10, verbose=1):
    models_params = {
        "Logistic": {
            "model": LogisticRegression(
                solver="qn", random_state=42, class_weight="balanced"
            ),
            "param_grid": {
                "C": np.logspace(-4, 4, 5),  # C参数的范围，使用对数间隔
                "penalty": ["l1", "l2"],  # 正则化类型
            },
        },
        "Lasso": {
            "model": Lasso(),
            "param_grid": {
                "alpha": np.logspace(-4, 4, 10),
            },
        },
        "ElasticNet": {
            "model": ElasticNet(),
            "param_grid": {
                "alpha": np.logspace(-4, 4, 5),
                "l1_ratio": np.linspace(0, 1, 2),
            },
        },
        # "RandomForest": {
        #     "model": RandomForestClassifier(),
        #     "param_grid": {"n_estimators": range(10, 101, 10)},
        # },
    }
    if method_list is not None:
        models_params = {k: v for k, v in models_params.items() if k in method_list}

    train_df = train_df[[y_var] + X_var].copy().dropna()
    test_df = test_df[[y_var] + X_var].copy().dropna()
    train_df[y_var] = train_df[y_var].astype(int)
    test_df[y_var] = test_df[y_var].astype(int)

    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    X_train = train_df[X_var]
    y_train = train_df[y_var]
    X_val = val_df[X_var]
    y_val = val_df[y_var]

    X_test = test_df[X_var]
    y_test = test_df[y_var]
    print(
        f"train shape: {X_train.shape}, val shape is {X_val.shape}, test shape is {X_test.shape}"
    )
    best_models = []

    for model_name, mp in models_params.items():
        # if model_name == "RandomForest":
        #     best_model = RandomForestClassifier(verbose=verbose)
        #     best_model.fit(X_train.values, y_train.values)
        #     auc = roc_auc_score(y_val, best_model.predict(X_val.values))
        #     bset_params = None  # no params for RandomForest

        # else:
        if model_name == "Logistic":
            scorer = make_scorer(roc_auc_score, needs_proba=True)
        else:
            scorer = make_scorer(roc_auc_score)

        grid_search = GridSearchCV(
            mp["model"], mp["param_grid"], scoring=scorer, cv=cv, verbose=verbose
        )
        grid_search.fit(X_train.values, y_train.values)

        best_model = grid_search.best_estimator_
        bset_params = grid_search.best_params_

        if model_name == "Logistic":
            auc = roc_auc_score(y_val, best_model.predict_proba(X_val.values)[:, 1])
        else:
            auc = roc_auc_score(y_val, best_model.predict(X_val.values))
        print(f"model: {model_name}\tBest parameters: {bset_params}, with auc: {auc}")
        best_models.append((model_name, best_model, grid_search, auc))

    ## select the currently best
    # print(best_models)

    # 还原原始的train_df
    train_df = pd.concat([train_df, val_df], axis=0)
    X_train = train_df[X_var]
    y_train = train_df[y_var]

    best_mdoels = list(sorted(best_models, key=lambda x: x[-1], reverse=True))
    best_model_name, best_model, *_ = best_mdoels[0]

    if best_model_name == "Logistic":
        train_pred = best_model.predict_proba(X_train.values)[:, 1]

        test_pred = best_model.predict_proba(X_test.values)[:, 1]
    else:
        train_pred = best_model.predict(X_train.values)
        val_pred = best_model.predict(X_val.values)
        test_pred = best_model.predict(X_test.values)

    train_df[f"{y_var}_pred"] = train_pred

    test_df[f"{y_var}_pred"] = test_pred

    train_auc = roc_auc_score(y_train, train_pred)
    test_auc = roc_auc_score(y_test, test_pred)

    train_metrics = {
        "train_auc": train_auc,
    }
    test_metrics = {
        "test_auc": test_auc,
    }
    return best_model, train_metrics, test_metrics, train_df, test_df, best_mdoels
