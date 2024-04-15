from cuml import LogisticRegression, Lasso, Ridge, ElasticNet
from cuml.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
from tqdm.rich import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
from ppp_prediction.corr import cal_binary_metrics_bootstrap
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from ppp_prediction.utils import DataFramePretty
import pickle


def lasso_select_model(
    train_df,
    test_df,
    features,
    label,
    n_bootstrap=500,
    threads=4,
    cv=5,
    # bins=9,
    save_dir="./output",
    name="select",
):
    current_save_path = save_dir
    key = name
    train_file = train_df
    test_file = test_df
    method = "Lasso"

    Path(current_save_path).mkdir(parents=True, exist_ok=True)
    current_save_pkl_path = f"{save_dir}/{key}.pkl"

    Path(current_save_pkl_path).parent.mkdir(parents=True, exist_ok=True)

    # step1 bootstrap training
    (
        model,
        train_metrics,
        test_metrics,
        train_imputed_data,
        test_imputed_data,
    ) = fit_best_model_bootstrap(
        train_df=train_file,
        test_df=test_file,
        X_var=features,
        y_var=label,
        method_list=method,
        cv=cv,
        n_resample=n_bootstrap,
        n_jobs=threads,
    )

    test_metrics = cal_binary_metrics_bootstrap(
        test_imputed_data[label],
        test_imputed_data[f"{label}_pred"],
        ci_kwargs=dict(n_resamples=1000),
    )
    all_obj = {
        "model": model,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }

    try:
        DataFramePretty(pd.Series(test_metrics).to_frame()).show()
    except:
        pass

    pickle.dump(all_obj, open(f"{current_save_pkl_path}", "wb"))

    # step2 Sensitivity analysis
    ## plot
    try:
        model._plot._plot_top_k_features()
        plt.savefig(f"{current_save_path}/top_k_features.png", dpi=400)
        plt.clf()
        model._show_models_coeffients()
        plt.savefig(f"{current_save_path}/coeffients.png", dpi=400)
        plt.clf()
    except:
        pass
    try:
        import seaborn as sns

        sns.kdeplot(
            model.weights_dist_df["percent_of_nonZero_coefficients"],
            shade=True,
            color="r",
        )
        plt.savefig(f"{current_save_path}/percent_of_nonZero_coefficients.png", dpi=400)
        plt.clf()
    except:
        pass
    ## fit model and get the best cutoff
    cutoff_bins = list(
        range(10, 100, 10)
    )  # 9个区间：>10, >20, >30, >40, >50, >60, >70, >80, >90
    cutoff_models = {}

    cuttoff_model_savedir = f"{current_save_path}/cutoff_models"
    Path(cuttoff_model_savedir).mkdir(parents=True, exist_ok=True)

    for cutoff in cutoff_bins:
        cutoff_features = model.weights_dist_df[
            model.weights_dist_df["percent_of_nonZero_coefficients"] > cutoff
        ].index.tolist()
        print(f"cutoff: {cutoff}, features num: {len(cutoff_features)}")
        (
            cutoff_model,
            cutoff_model_train_metrics,
            cutoff_model_test_metrics,
            cutoff_train_imputed_data,
            cutoff_test_imputed_data,
            _,
        ) = fit_best_model(
            train_df=train_file,
            test_df=test_file,
            X_var=cutoff_features,
            y_var=label,
            method_list=method,
            cv=cv,
        )
        cutoff_model_test_metrics["cutoff"] = cutoff
        cutoff_models[cutoff] = {
            "model": cutoff_model,
            "test_metrics": cutoff_model_test_metrics,
        }

        pickle.dump(
            cutoff_model, open(f"{cuttoff_model_savedir}/cutoff_{cutoff}.pkl", "wb")
        )

    # step3 use the best cutoff to get the final proteins
    cutoff_model_test_metrics_list = [i["test_metrics"] for i in cutoff_models.values()]
    cutoff_model_test_metrics_df = pd.DataFrame(cutoff_model_test_metrics_list)
    ## plot
    try:
        fig, ax = plt.subplots()
        ax.errorbar(
            x=cutoff_model_test_metrics_df["cutoff"],
            y=cutoff_model_test_metrics_df["AUC"],
            yerr=[
                cutoff_model_test_metrics_df["AUC"]
                - cutoff_model_test_metrics_df["AUC_LCI"],
                cutoff_model_test_metrics_df["AUC_UCI"]
                - cutoff_model_test_metrics_df["AUC"],
            ],
            fmt="o",
            marker="o",
            mfc="black",
            color="black",
            capsize=4,
            label="AUC of cutoff model",
        )
        ax.set_xlabel("cutoff")
        ax.set_ylabel("AUC")
        plt.savefig(f"{current_save_path}/cutoff_AUC.png", dpi=400)
    except:
        pass

    ## get the best cutoff
    best_cutoff = cutoff_model_test_metrics_df.sort_values("AUC", ascending=False).iloc[
        0
    ]["cutoff"]
    print(f"best cutoff is {best_cutoff}")
    print(f"{cuttoff_model_savedir}/cutoff_{best_cutoff}.pkl")


def fit_best_model(train_df, test_df, X_var, y_var, method_list=None, cv=10, verbose=1):
    models_params = {
        "Logistic": {
            "model": LogisticRegression(
                solver="qn", random_state=42, class_weight="balanced"
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
                "alpha": np.logspace(-4, 4, 10),
                "l1_ratio": np.linspace(0, 1, 10),
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
        rf = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", mp["model"]),
            ]
        )
        params_dict = {f"model__{k}": v for k, v in mp["param_grid"].items()}
        grid_search = GridSearchCV(
            rf, params_dict, scoring=scorer, cv=cv, verbose=verbose
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
    test_metrics = cal_binary_metrics_bootstrap(
        y=y_test, y_pred=test_pred, ci_kwargs=dict(n_resamples=200)
    )
    # test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}
    try:
        
        best_model['model'].feature_names_in_ = X_var # add feature names to model
    except: 
        pass 
    
    return best_model, train_metrics, test_metrics, train_df, test_df, best_mdoels


class EnsembleModel(object):
    def __init__(self, models, coef_name=None, model_name_list=None):
        self.models = models

        if coef_name is None:
            if hasattr(self.models[0], "feature_names_in_"):
                coef_name = self.models[0].coef_name
            else:
                raise ValueError("coef_name should be provided")
        else:
            coef_name = coef_name
        self.features = coef_name if coef_name else self.models["model"].coef_name

        self.model_name_list = (
            model_name_list
            if model_name_list
            else [f"model_{i}" for i in range(len(self.models))]
        )

        self.res = self._init_coeffeients_df()
        self._init_weights_dist()

    def _init_coeffeients_df(self):

        res = pd.concat(
            [
                pd.DataFrame(
                    (
                        model_each.coef_
                        if hasattr(model_each, "coef_")
                        else model_each["model"].coef_
                    ),
                    index=self.features,
                    columns=["coefficients"],
                ).sort_values("coefficients", ascending=False)
                for model_each in self.models
            ],
            axis=1,
        )
        res.columns = [f"model_{i}" for i in range(len(self.model_name_list))]
        return res

    def _init_weights_dist(self):
        if self.res is None:
            self.res = self._init_coeffeients_df()
        res = self.res.loc[self.features, :]

        percent_of_nonZero_coefficients = (
            (res != 0).sum(axis=1) * 100 / len(res.columns)
        )
        mean_coefficients = res.mean(axis=1)
        weights_dist_df = pd.DataFrame(
            [percent_of_nonZero_coefficients, mean_coefficients],
            index=["percent_of_nonZero_coefficients", "mean_coefficients"],
        ).T
        weights_dist_df["abs_mean_coefficients"] = weights_dist_df[
            "mean_coefficients"
        ].abs()
        self.weights_dist_df = weights_dist_df

    def _plot_top_k_features(self, k=10, pallete="viridis", ax=None):
        """
        plot top k features
        """
        if not hasattr(self, "weights_dist_df"):
            self._init_weights_dist()
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, k))

        top_k_features = self.weights_dist_df.sort_values(
            by=["mean_coefficients"],
            ascending=False,
        )
        plt_data = self.res.loc[
            [*top_k_features.index[:k], *top_k_features.index[-k:]], :
        ]
        plt_data = plt_data.reset_index(drop=False).melt(id_vars="index")

        sns.boxplot(
            data=plt_data,
            y="index",
            x="value",
            showfliers=False,
            ax=ax,
            palette=pallete,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=10)
        ax.set_xticks([0])
        ax.grid(axis="x", linestyle="--", alpha=1, linewidth=2, color="red")
        ax.set_xlabel("Mean of Coefficients")
        ax.set_ylabel("Features")
        ax.set_title(f"Top {k} Features")
        return ax

    def _show_models_coeffients(self, axes=None, color="#d67b7f", top=5):
        """
        res:
            model1 model2
        SOST xx yy
        BGN xx yy


        """
        if self.res is None:
            self.res = self._init_coeffeients_df()
        res = self.res.loc[self.features, :]

        if axes is None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        else:
            assert len(axes) == 2, "axes should be a list of length 2"
            ax1, ax2 = axes

        percent_of_nonZero_coefficients = (
            (res != 0).sum(axis=1) * 100 / len(res.columns)
        )
        mean_coefficients = res.mean(axis=1)
        plt_data = pd.DataFrame(
            [percent_of_nonZero_coefficients, mean_coefficients],
            index=["percent_of_nonZero_coefficients", "mean_coefficients"],
        ).T
        plt_data["abs_mean_coefficients"] = plt_data["mean_coefficients"].abs()

        # ax1
        sns.scatterplot(
            x=percent_of_nonZero_coefficients,
            y=mean_coefficients,
            size=mean_coefficients,
            sizes=(20, 400),
            legend=False,
            edgecolor="black",
            ax=ax1,
            color=color,
        )
        ax1.plot([0, 100], [0, 0], "k--", lw=3, color="grey")
        ax1.set_xlim(-1, 105)
        ax1.set_xlabel("percent of non-zero coefficients")
        ax1.set_ylabel("mean nonzero coefficients")
        sorted_plt_data = (
            plt_data.sort_values(
                by=["percent_of_nonZero_coefficients", "abs_mean_coefficients"],
                ascending=False,
            )
            .iloc[:top, :]
            .index
        )
        for i, txt in enumerate(sorted_plt_data):
            # ax1.annotate(txt, (sorted_plt_data.iloc[i, 0], sorted_plt_data.iloc[i, 1]))
            ax1.text(
                plt_data.loc[txt, "percent_of_nonZero_coefficients"],
                plt_data.loc[txt, "mean_coefficients"],
                txt,
                ha="right",
                fontsize=8,
                color="black",
            )

        # ax2
        absolute_mean_coefficients = mean_coefficients.abs().sort_values(ascending=True)
        sns.barplot(
            y=absolute_mean_coefficients,
            x=absolute_mean_coefficients.index,
            ax=ax2,
            color=color,
        )
        ax2.set_ylabel("absolute mean coefficients")
        ax2.set_xlabel("")
        xticks = ax2.get_xticklabels()
        if len(xticks) > 100:
            ax2.set_xticks([""] * len(xticks))
        else:
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
        if axes is None:
            # fig.tight_layout()
            return ax1, ax2

    def predict(self, data):
        preds = []

        # check all feature in data 
        data = data.loc[:, self.features].copy()

        

        for model in self.models:
            if hasattr(model, "predict_proba"):
                preds.append(model.predict_proba(data)[:, 1])
            else:
                preds.append(model.predict(data))
        return np.mean(preds, axis=0)

    def predict_df(self, data):

        data[f"pred_{self.label}"] = self.predict(data)
        return data


def fit_best_model_bootstrap(
    train_df,
    test_df,
    X_var,
    y_var,
    method_list=None,
    cv=10,
    verbose=1,
    n_resample=100,
    n_jobs=4,
):

    if n_jobs == 1:
        random_stats = [i for i in np.random.randint(0, 100000, n_resample)]
        res = []
        for i in tqdm(random_stats):
            train_df_sample = train_df.sample(frac=1, replace=True, random_state=i)
            best_model, *_ = fit_best_model(
                train_df_sample, test_df, X_var, y_var, method_list, cv, verbose
            )
            res.append(best_model)

    else:
        from joblib import Parallel, delayed

        print(f"n_jobs: {n_jobs}")
        # random_stats = [i for i in np.random.randint(0, 100000, n_resample)]
        random_stats = list(range(n_resample))

        def fit_best_model_modified(*args):
            best_model, *_ = fit_best_model(*args)
            return best_model

        res = Parallel(n_jobs=n_jobs)(
            delayed(fit_best_model_modified)(
                train_df.sample(frac=1, replace=True, random_state=i),
                test_df,
                X_var,
                y_var,
                method_list,
                cv,
                verbose,
            )
            for i in tqdm(random_stats)
        )

    model = EnsembleModel(res, coef_name=X_var, model_name_list=None)

    train_df[f"{y_var}_pred"] = model.predict(train_df[X_var].values)
    test_df[f"{y_var}_pred"] = model.predict(test_df[X_var].values)
    train_metrics = cal_binary_metrics_bootstrap(
        y=train_df[y_var].values,
        y_pred=train_df[f"{y_var}_pred"].values,
        ci_kwargs=dict(n_resamples=200),
    )
    test_metrics = cal_binary_metrics_bootstrap(
        y=test_df[y_var].values,
        y_pred=test_df[f"{y_var}_pred"].values,
        ci_kwargs=dict(n_resamples=200),
    )
    test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}
    return model, train_metrics, test_metrics, train_df, test_df