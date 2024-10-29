from cuml import LogisticRegression, Lasso, Ridge, ElasticNet
from cuml.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
from tqdm.rich import tqdm
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from ppp_prediction.utils import DataFramePretty
import pickle


from sklearn.pipeline import Pipeline
from typing import Union, List
from sklearn.utils._metadata_requests import process_routing
import numpy as np
from ppp_prediction.model_v2.models import fit_best_model_v2
from ppp_prediction.metrics import cal_binary_metrics

import logging
from functools import partial
from ppp_prediction.model_v2.models import (
    fit_lightgbm,
    fit_xgboost,
    fit_best_model_v2,
    get_predict_v2_from_df,
)
import time
# from rich.
from tqdm.rich import tqdm
import warnings


warnings.filterwarnings("ignore")



class EnsembleModel(object):
    def __init__(self, models, cov=None, coef_name=None, model_name_list=None):
        self.models = models

        if coef_name is None:
            if hasattr(self.models[0], "feature_names_in_"):
                coef_name = self.models[0].feature_names_in_.tolist()
            else:
                raise ValueError("coef_name should be provided")
        else:
            coef_name = coef_name

        if cov:
            if isinstance(cov, str):
                cov = [cov]
            # print(f"will drop cov: {','.join(cov)}")
            # coef_name = list(set(coef_name) - set(cov))
        self.features = coef_name
        self.feature_names_in_ = coef_name
        self.cov = cov

        self.model_name_list = (
            model_name_list
            if model_name_list
            else [f"model_{i}" for i in range(len(self.models))]
        )

        self.res = self._init_coeffeients_df()
        self._init_weights_dist()

    def reinit(self, new_coef=None, cov=None):
        self.__init__(
            self.models,
            coef_name=new_coef,
            cov=cov,
            model_name_list=self.model_name_list,
        )

    def __repr__(self) -> str:
        return str(self.models)

    def _init_coeffeients_df(self):

        res = []
        for model_each in self.models:
            coef_ = (
                model_each.coef_
                if hasattr(model_each, "coef_")
                else model_each["model"].coef_
            )
            if hasattr(model_each, "feature_names_in_"):
                feature_names_in_ = model_each.feature_names_in_
            elif hasattr(model_each["model"], "feature_names_in_"):
                feature_names_in_ = model_each["model"].feature_names_in_
            else:
                raise ValueError(f"No feature_names_in found in {model_each}")
            # feature_names_in_
            choice_idx = np.isin(feature_names_in_, self.features)
            coef_ = coef_[choice_idx]
            feature_names_in_ = feature_names_in_[choice_idx]
            res_df = pd.DataFrame(coef_, index=feature_names_in_)
            res_df.columns = ["coefficients"]
            res_df.sort_values("coefficients", ascending=False)
            res.append(res_df)

        res = pd.concat(res, axis=1)
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

    def _plot_top_k_features(self, k=10, pallete="viridis", ax=None, exclude=None):
        """
        plot top k features
        """
        if not hasattr(self, "weights_dist_df"):
            self._init_weights_dist()
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, k))

        exclude = self.cov if exclude is None else exclude + self.cov
        if isinstance(exclude, str):
            exclude = [exclude]

        if exclude is not None:
            plt_data = self.res.loc[self.res.index.difference(exclude), :]

            top_k_features = self.weights_dist_df
            top_k_features = top_k_features.loc[
                top_k_features.index.difference(exclude), :
            ].sort_values(
                by=["mean_coefficients"],
                ascending=False,
            )
        else:
            plt_data = self.res
            top_k_features = self.weights_dist_df.sort_values(
                by=["mean_coefficients"],
                ascending=False,
            )

        plt_data = plt_data.loc[
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

    def _show_models_coeffients(self, axes=None, color="#d67b7f", top=5, exclude=None):
        """
        res:
            model1 model2
        SOST xx yy
        BGN xx yy


        """
        if self.res is None:
            self.res = self._init_coeffeients_df()
        res = self.res.loc[self.features, :]
        if self.cov is not None:
            exclude = self.cov if exclude is None else exclude + self.cov
            if isinstance(exclude, str):
                exclude = [exclude]
            res = res.loc[res.index.difference(exclude), :]

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

    def coef_barplot(
        self,
        cmap="RdBu_r",
        k=10,
        ax=None,
        errorbar_kwargs=None,
        scatter_kwargs=None,
        exclude=["age", "sex"],
    ):
        from adjustText import adjust_text
        from scipy import stats

        if errorbar_kwargs is None:
            errorbar_kwargs = {}
        if scatter_kwargs is None:
            scatter_kwargs = {}

        plt_data = self.res.copy()

        def cal_ci(x):
            mean_x = x.mean()
            scale = stats.sem(x)
            ci_low, ci_high = stats.t.interval(
                0.95, len(x) - 1, loc=mean_x, scale=scale
            )
            return {"mean": mean_x, "ci_low": ci_low, "ci_high": ci_high}

        # drop age sex
        if self.cov is not None:
            exclude = self.cov if exclude is None else exclude + self.cov
            if isinstance(exclude, str):
                exclude = [exclude]

            plt_data = plt_data.loc[[i not in exclude for i in plt_data.index.tolist()]]

        plt_data = plt_data.apply(
            lambda x: pd.Series(cal_ci(x)),
            axis=1,
        )
        plt_data = plt_data.sort_values("mean", ascending=False).reset_index(
            drop=False, names="feature"
        )
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        plt_data["error_low"] = plt_data["mean"] - plt_data["ci_low"]
        plt_data["error_high"] = plt_data["ci_high"] - plt_data["mean"]

        # 绘制误差线

        ax.errorbar(
            x=plt_data.index,
            y=plt_data["mean"],
            yerr=[
                plt_data["mean"] - plt_data.ci_low,
                plt_data.ci_high - plt_data["mean"],
            ],
            fmt=errorbar_kwargs.pop("fmt", "none"),  # 不使用标记
            lw=errorbar_kwargs.pop("lw", 1),
            capsize=errorbar_kwargs.pop("capsize", 2),
            ecolor=errorbar_kwargs.pop("ecolor", "lightgrey"),  # 将误差线设置为浅灰色
            **errorbar_kwargs,
        )

        # 使用scatter添加颜色渐变的散点
        # 计算每个点的颜色
        colors = plt_data["mean"]
        min_value = max(abs(colors.min()), abs(colors.max()))
        norm = plt.Normalize(-min_value, min_value)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        sc = ax.scatter(
            plt_data.index,
            plt_data["mean"],
            c=colors,
            cmap=cmap,
            s=scatter_kwargs.pop("s", 5),
            # edgecolor="k",
            zorder=scatter_kwargs.pop("zorder", 3),
            **scatter_kwargs,
        )
        cb = plt.colorbar(sm, ax=ax)

        # 设置标题和轴标签
        ax.set_title(
            "Mean Coefficient of bootstrap model", fontsize=16, fontweight="bold"
        )
        ax.set_xlabel(
            "",
        )
        ax.set_ylabel("Mean Coefficient", fontsize=14)
        ax.set_yticks([-min_value / 2, 0, min_value / 2])
        ax.set_xticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # 增加网格线
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

        texts = [
            ax.text(
                idx,
                row["mean"] + row["error_high"],
                f"{row['feature']}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            for idx, row in plt_data.head(k).iterrows()
        ] + [
            ax.text(
                idx,
                row["mean"] - row["error_low"],
                f"{row['feature']}",
                ha="center",
                va="top",
                fontsize=8,
            )
            for idx, row in plt_data.tail(k).iterrows()
        ]
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="->", lw=0.5))

        return ax

    def predict(self, data, exclude=None, method="mean"):
        preds = []

        # check all feature in data
        data = data.loc[:, self.features].copy()
        if self.cov:
            exclude = self.cov

        for model in self.models:
            if exclude:  # 去除cov效应
                preds.append(
                    get_predict(model, data, x_var=self.features, exclude=exclude)
                )
            else:
                if hasattr(model, "predict_proba"):
                    preds.append(model.predict_proba(data)[:, 1])
                else:
                    preds.append(model.predict(data))

        if hasattr(self, "weight_model") and method == "weight_model":
            return self.weight_model.predict(np.array(preds).T)
        else:
            return np.mean(preds, axis=0)

    def fit_ensemble_weight(
        self,
        train_data,
        test_data,
        label="label",
    ):
        """
        fit ensemble weight
        """

        train_data = (
            train_data.loc[:, self.features + [label]]
            .copy()
            .dropna()
            .reset_index(drop=True)
        )
        test_data = (
            test_data.loc[:, self.features + [label]]
            .copy()
            .dropna()
            .reset_index(drop=True)
        )

        train_dict = {}
        test_dict = {}

        exclude = self.cov

        for model_name, model in zip(self.model_name_list, self.models):
            if hasattr(model, "predict_proba"):
                train_dict[model_name] = model.predict_proba(train_data[self.features])[
                    :, 1
                ]
                test_dict[model_name] = model.predict_proba(test_data[self.features])[
                    :, 1
                ]

            else:
                if exclude:
                    train_dict[model_name] = get_predict(
                        model, train_data[self.features], exclude=exclude
                    )
                    test_dict[model_name] = get_predict(
                        model, test_data[self.features], exclude=exclude
                    )
                else:
                    train_dict[model_name] = model.predict(train_data[self.features])
                    test_dict[model_name] = model.predict(test_data[self.features])

        train_df = pd.DataFrame(train_dict)
        test_df = pd.DataFrame(test_dict)
        # print(test_data.columns)
        # print(test_df)
        train_df["label"] = train_data[label]
        test_df["label"] = test_data[label]

        X_var = self.model_name_list
        print(f"train shape: {train_df.shape}, test shape is {test_df.shape}")

        weight_model, train_metrics, test_metrics, *_ = fit_best_model(
            train_df=train_df,
            test_df=test_df,
            X_var=X_var,
            y_var="label",
            method_list=["Lasso"],
            cv=5,
        )
        # weight_model.features = X_var
        self.weight_model = weight_model
        return weight_model, train_metrics, test_metrics
    




def fit_best_model_bootstrap_v2(
    train,
    xvar,
    label,
    method_list=["Lasso"],
    test=None,
    cv=10,
    verbose=1,
    n_resample=100,
    n_jobs=4,
    save_dir=None,
    engine="cuml",
    test_size=0.3,
):
    if test is None:
        train, test = train_test_split(train, test_size=test_size)
    if Path(save_dir).exists():
        find_models = list(Path(save_dir).rglob("*.pkl"))
        res = [pd.read_pickle(i) for i in find_models]
        if len(find_models) == n_resample:
            logging.info(f"find {n_resample} models in {save_dir} and will load")
        elif len(find_models) > 0 and len(find_models) < n_resample:
            retrain_n = n_resample - len(find_models)
            logging.warning(
                f"find {len(find_models)} models in {save_dir}, {len(res)} less than {n_resample}, will retrain {retrain_n} models"
            )
            n_resample = retrain_n
    else:
        res = []  
        n_resample = n_resample

    if n_resample > 0:
        res_trained = [] 
        if n_jobs == 1:
            random_stats = [i for i in np.random.randint(0, 100000, n_resample)]
            for i in tqdm(random_stats):
                train_df_sample = train.sample(frac=1, replace=True, random_state=i)
                best_model, *_ = fit_best_model_v2(
                    train_df_sample,
                    xvar,
                    label,
                    None,
                    method_list,
                    cv,
                    verbose,
                    save_dir=f"{save_dir}/{i}.pkl" if save_dir else None,
                    engine=engine,
                )
                res_trained.append(best_model)

        else:
            from joblib import Parallel, delayed

            print(f"n_jobs: {n_jobs}")
            # random_stats = [i for i in np.random.randint(0, 100000, n_resample)]
            random_stats = list(range(n_resample))

            def fit_best_model_modified(*args):
                try:
                    best_model, *_ = fit_best_model_v2(*args)
                    return best_model
                except:
                    return None

            res_trained = Parallel(n_jobs=n_jobs)(
                delayed(fit_best_model_modified)(
                    train.sample(frac=1, replace=True, random_state=i),
                    xvar,
                    label,
                    None,
                    method_list,
                    cv,
                    verbose,
                    f"{save_dir}/{i}.pkl" if save_dir else None,
                    engine,
                )
                for i in tqdm(random_stats)
            )

        total_success = [i for i in res_trained if i is not None]

        if total_success == 0:
            raise ValueError("No model is fitted successfully")
        else:
            print(f"total success: {len(total_success)} with {n_resample} resamples")

        # add success model to res
        res_trained = [i for i in res_trained if i is not None]
        res.extend(res_trained)

    model = EnsembleModel(res, coef_name=xvar, model_name_list=None)

    train[f"{label}_pred"] = model.predict(train[xvar])
    to_cal_train = train[[label, f"{label}_pred"]].copy().dropna()
    train_metrics = cal_binary_metrics(
        y=to_cal_train[label].values,
        y_pred=to_cal_train[f"{label}_pred"].values,
        ci=True,
    )

    test[f"{label}_pred"] = model.predict(test[xvar])
    to_cal_test = test[[label, f"{label}_pred"]].copy().dropna()
    test_metrics = cal_binary_metrics(
        y=to_cal_test[label].values, y_pred=to_cal_test[f"{label}_pred"].values, ci=True
    )
    test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}
    return model, train_metrics, test_metrics, train, test






def BootstrapLassoSelectModelAndFit(
    train,
    xvar,
    label,
    test=None,
    methods=None,
    cutoff_list=None,
    threads=1,
    save_dir="./BootstrapLassoSelectModelAndFit",
    bootstrap_engine="cuml",
    bootstrap_n_resample=100,
):
    """

    save_dir/
        bootstrap/
        methods/
            methods1/
            methods2/
        comapre.csv

    methods should be a list or dict

    if a dict like below
    {
        "Method1", method_fn1,
        ............
    },

    will use method_fn1 to fit model with diff cutoff


    """
    train = train.copy()
    if test is None:
        train, test = train_test_split(train, test_size=0.3, random_state=42)
    else:
        test = test.copy()

    if cutoff_list is None:
        cutoff_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # non-zero percent

    # Define method_fn
    supported_methods = {
        "Lasso": partial(fit_best_model_v2, method_list=["Lasso"], engine="cuml",cv =3),
        # "Ridge": partial(fit_best_model_v2, method_list=["Ridge"], engine="sklearn"),
        # "ElasticNet": partial(
        #     fit_best_model_v2, method_list=["ElasticNet"], engine="sklearn"
        # ),
        "xgboost": partial(fit_xgboost, tuning=True, tune_config={"n_cpus": threads}),
        "lightgbm": partial(
            fit_lightgbm, tuning=True, tune_config={"n_cpus": threads}
        ),
    }
    if isinstance(methods, str):
        methods = [methods]

    if methods is None:
        methods_dict = {"Lasso": {fit_best_model_v2}}

    elif isinstance(methods, list):
        methods_dict = {}
        for method in methods:
            if method not in supported_methods:
                logging.warning(f"method {method} is not supported")
                methods.remove(method)
            else:
                methods_dict[method] = supported_methods[method]

    elif isinstance(methods, dict):
        methods_dict = {}
        for method_name, method_fn in methods.items():
            if hasattr(method_fn, "__call__"):
                methods_dict[method_name] = method_fn
            else:
                logging.warning(f"method {method_name} is not a function")

    bootstrap_save_dir = Path(save_dir) / "bootstrap"
    bootstrap_save_dir.mkdir(parents=True, exist_ok=True)

    fig_save_dir = Path(save_dir) / "figs"
    fig_save_dir.mkdir(parents=True, exist_ok=True)
    # step1 fit best model
    bootstrap_model_dir = save_dir / "bootstrap_model.pkl"
    if bootstrap_model_dir.exists():
        bootsrap_model = pickle.load(open(bootstrap_model_dir, "rb"))
    else:
        bootstrap_time_start = time.time()

        bootsrap_model, *_ = fit_best_model_bootstrap_v2(
            train,
            xvar,
            label,
            n_resample=bootstrap_n_resample,
            n_jobs=threads,
            engine=bootstrap_engine,
            save_dir=bootstrap_save_dir,
        )
        bootstrap_time_end = time.time()
        bootstrap_time = bootstrap_time_end - bootstrap_time_start
        logging.info(f"bootstrap model is fitted in {bootstrap_time} seconds")
        
        ## save bootstrap model 
        pickle.dump(bootsrap_model, open(bootstrap_model_dir, "wb"))

    ## plot
    try:
        bootsrap_model._plot._plot_top_k_features()
        plt.savefig(f"{fig_save_dir}/top_k_features.png", dpi=400)
        plt.clf()
    except Exception as e:
        logging.warning(f"plot top k features failed: {e}")
        pass 
    try:
        bootsrap_model._show_models_coeffients()
        plt.savefig(f"{fig_save_dir}/coeffients.png", dpi=400)
        plt.clf()
    except Exception as e:
        logging.warning(f"plot coeffients failed: {e}")
        pass
    try:
        import seaborn as sns

        sns.kdeplot(
            bootsrap_model.weights_dist_df["percent_of_nonZero_coefficients"],
            shade=True,
            color="r",
        )
        plt.savefig(f"{fig_save_dir}/percent_of_nonZero_coefficients.png", dpi=400)
        plt.clf()
    except Exception as e:
        logging.warning(f"plot percent_of_nonZero_coefficients failed: {e}")
        pass

    # step2 fit model for each cut off
    cutoff_bins = cutoff_list + [
        "ALL"
    ]  # 9个区间：>10, >20, >30, >40, >50, >60, >70, >80, >90

    method_trained_dict = {("bootstrap", "ALL"): bootsrap_model}
    try:
        time_dict = {("bootstrap", "ALL"):bootstrap_time}
    except:  # noqa: E722
        time_dict = {("bootstrap", "ALL"):np.nan}
    for cutoff in cutoff_bins:
        if cutoff == "ALL":
            cutoff_features = bootsrap_model.weights_dist_df.index.tolist()
        else:
            cutoff_features = bootsrap_model.weights_dist_df[
                bootsrap_model.weights_dist_df["percent_of_nonZero_coefficients"]
                >= cutoff
            ].index.tolist()
        if len(cutoff_features) == 0:
            logging.warning(f"no feature is selected with cutoff {cutoff}")
            continue

        for method_name, method_fn in methods_dict.items():
            c_cutoff_method_save_dir = Path(save_dir) / f"{method_name}/{cutoff}"
            c_cutoff_method_save_dir.mkdir(parents=True, exist_ok=True)
            c_cutoff_method_model_dir = c_cutoff_method_save_dir / f"{method_name}.pkl"
            if not c_cutoff_method_model_dir.exists():
                logging.info(f"fitting {method_name} with cutoff {cutoff}")
                method_fn_time_start = time.time()
                return_obj = method_fn(
                    train,
                    cutoff_features,
                    label,
                    test=None,
                )
                model, *_ = return_obj

                method_fn_time_end = time.time()
                method_fn_time = method_fn_time_end - method_fn_time_start
                logging.info(
                    f"{method_name} with cutoff {cutoff} is fitted in {method_fn_time} seconds"
                )
                method_trained_dict[(method_name, cutoff)] = model
                time_dict[(method_name, cutoff)] = method_fn_time
                pickle.dump(
                    model, open(c_cutoff_method_model_dir, "wb")
                )
            else:
                logging.info(f"load {c_cutoff_method_model_dir}")
                model = pickle.load(open(c_cutoff_method_model_dir, "rb"))
                method_trained_dict[(method_name, cutoff)] = model
                time_dict[(method_name, cutoff)] = np.nan

    # step3 compare

    compare_list = []
    logging.info("start to cal compare metrics")
    for (model_name, cutoff), model in tqdm(method_trained_dict.items(), total=len(method_trained_dict), desc="Runing"):
        col_name = "_".join([model_name, str(cutoff)])
        if hasattr(model, "feature_names_in_"):
            cutoff_xvar = model.feature_names_in_
        elif hasattr(model, "feature_name"):
            # this is lgb booster may have; 
            # TODO: 统一接口
            cutoff_xvar = model.feature_name()
        # TODO: check if cutoff_xvar is np.ndarray ,and turn into list
        if not isinstance(cutoff_xvar, list):
            cutoff_xvar = cutoff_xvar.tolist()

        train[col_name] = model.predict(train[cutoff_xvar])
        test[col_name] = model.predict(test[cutoff_xvar])

        to_cal_test = test[[label, col_name]].copy().dropna()

        test_metrics = cal_binary_metrics(
            to_cal_test[label],
            to_cal_test[col_name],
            ci=True,
        )
        test_metrics["model_name"] = model_name
        test_metrics["cutoff"] = cutoff
        test_metrics["fitting_time"] = time_dict[(model_name, cutoff)]
        compare_list.append(test_metrics)

    compare_df = pd.DataFrame(compare_list)
    compare_df.to_csv(Path(save_dir) / "compare.csv", index=False)

    pickle.dump(
        method_trained_dict, open(Path(save_dir) / "method_trained_dict.pkl", "wb")
    )
    return method_trained_dict, compare_df