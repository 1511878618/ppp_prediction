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

    
from sklearn.pipeline import Pipeline
from typing import Union, List
from sklearn.utils._metadata_requests import process_routing
import numpy as np

def log_likelihood(y_true, y_pred, n):
    """
    Calculate the log-likelihood of a regression model.

    Parameters:
    - y_true: array, true target values
    - y_pred: array, predicted target values by the model
    - n: int, number of observations

    Returns:
    - ll: float, the log-likelihood value
    """
    residuals = y_true - y_pred
    residuals_squared_mean = np.mean(residuals**2)
    ll = -0.5 * n * (1 + np.log(2 * np.pi) + np.log(residuals_squared_mean))
    return ll


def calculate_aic_bic(y_true, y_pred, n, k):
    """
    Calculate AIC and BIC for a regression model.

    Parameters:
    - y_true: array, true target values
    - y_pred: array, predicted target values by the model
    - n: int, number of observations
    - k: int, number of features in the model

    Returns:
    - A tuple containing:
      - AIC: float, Akaike Information Criterion
      - BIC: float, Bayesian Information Criterion
    """
    ll = log_likelihood(y_true, y_pred, n)
    aic = -2 * ll + 2 * k
    bic = -2 * ll + k * np.log(n)
    return aic, bic
def get_predict(
    pipline: Pipeline,
    data,
    x_var: Union[List, str, None] = None,
    exclude: Union[List, str, None] = None,
    **params
):
    """
    用以对pipline的计算过程最后一步去除cov的权重
    
    """
    if x_var is not None :
        feature_names_in_ = x_var
    elif hasattr(pipline, "feature_names_in_"):
        feature_names_in_ = pipline.feature_names_in_
    elif hasattr(last_model, "feature_names_in_"):
        feature_names_in_ = last_model.feature_names_in_
    else:
        feature_names_in_ = data.columns.tolist() 

    data = data[feature_names_in_]

    if x_var is not None or exclude is not None:
        # currently only supported with scaler + model
        routed_params = process_routing(pipline, "predict", **params)
        Xt = data
        for _, name, transform in pipline._iter(with_final=False):
            Xt = transform.transform(Xt, **routed_params[name].transform)

        last_model = pipline.steps[-1][1]
        assert hasattr(last_model, "coef_") and hasattr(
            last_model, "intercept_"
        ), "pipline last step must have coef_ and intercept_ attributes."


        coef_ = last_model.coef_
        intercept_ = last_model.intercept_

        if exclude:
            exclude = [exclude] if isinstance(exclude, str) else exclude
            x_var = list(set(feature_names_in_) - set(exclude))
        x_var = [x_var] if isinstance(x_var, str) else x_var

        chosed_var_index = np.isin(feature_names_in_, x_var)

        coef_ = coef_[chosed_var_index]

        Xt = Xt[:, chosed_var_index]

        if coef_.ndim == 1:
            return Xt @ coef_ + intercept_
        else:
            return Xt @ coef_.T + intercept_
    else:


        return pipline.predict(data)


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
    name="EnsembleModel",
):
    current_save_path = save_dir
    key = name
    train_file = train_df
    test_file = test_df
    method = ["Lasso"]

    Path(current_save_path).mkdir(parents=True, exist_ok=True)
    current_save_pkl_path = f"{save_dir}/{key}_obj.pkl"

    Path(current_save_pkl_path).parent.mkdir(parents=True, exist_ok=True)

    bootstrap_model_dir = f"{current_save_path}/bootstrap_models"
    Path(bootstrap_model_dir).mkdir(parents=True, exist_ok=True)
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
        save_dir = bootstrap_model_dir,
    )
    to_cal_test = test_imputed_data[[label, f"{label}_pred"]].copy().dropna()
    test_metrics = cal_binary_metrics_bootstrap(
        to_cal_test[label],
        to_cal_test[f"{label}_pred"],
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
            save_dir = f"{cuttoff_model_savedir}/cutoff_{cutoff}.pkl"
        )
        cutoff_model_test_metrics["cutoff"] = cutoff
        aic,bic = calculate_aic_bic(y_true = test_file[label], y_pred = test_file[f"{label}_pred"], n = test_file.shape[0], k = len(cutoff_features))
        cutoff_model_test_metrics['AIC'], cutoff_model_test_metrics['BIC'] = aic, bic
        cutoff_model_test_metrics['total_params'] = len(cutoff_features)
        cutoff_model_test_metrics['total_non_zero_params'] = (cutoff_model['model'].coef_ != 0).sum()

        cutoff_models[cutoff] = {
            "model": cutoff_model,
            "test_metrics": cutoff_model_test_metrics,
        }

        pickle.dump(
            cutoff_model, open(f"{cuttoff_model_savedir}/cutoff_{cutoff}_obj.pkl", "wb")
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
    ## plot aic and bic
    try:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes = axes.flatten()
        aic_or_bic = ['AIC', 'BIC']
        for idx in range(2):
            ax = axes[idx]
            to_plot_metric = aic_or_bic[idx]
            sns.scatterplot(
                x=cutoff_model_test_metrics_df["cutoff"],
                y=cutoff_model_test_metrics_df[to_plot_metric],
                label=to_plot_metric,
                color = 'black',
                ax =ax 
            )
        fig.savefig(f"{current_save_path}/cutoff_AIC_BIC.png", dpi=400)
    except:
        pass

    ## get the best cutoff
    best_cutoff = cutoff_model_test_metrics_df.sort_values("AUC", ascending=False).iloc[
        0
    ]["cutoff"]
    print(f"best cutoff is {best_cutoff}")
    print(f"{cuttoff_model_savedir}/cutoff_{best_cutoff}.pkl")
    test_metrics['cutoff'] = "ensemble"
    pd.concat([cutoff_model_test_metrics_df, pd.DataFrame(test_metrics, index=[0])], axis=0).to_csv(f"{current_save_path}/test_metrics.csv", index=False)


def fit_best_model(train_df, test_df, X_var, y_var,method_list=None, cv=10, verbose=1,save_dir=None,y_type="bt"):
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
    if save_dir is not None and Path(save_dir).exists():
        best_model = pickle.load(open(save_dir, "rb"))
        if hasattr(best_model, "predict_proba"):
            best_model_name = "Logistic"
        else :
            best_model_name = "Lasso"
        best_models = None 
    else:
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
            # grid_search.fit(X_train.values, y_train.values)
            grid_search.fit(X_train, y_train)

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


        best_models = list(sorted(best_models, key=lambda x: x[-1], reverse=True))
        best_model_name, best_model, *_ = best_models[0]


    # 还原原始的train_df
    train_df = pd.concat([train_df, val_df], axis=0)
    X_train = train_df[X_var]
    y_train = train_df[y_var]

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
    # test_auc = roc_auc_score(y_test, test_pred)

    train_metrics = {
        "train_auc": train_auc,
    }
    test_metrics = cal_binary_metrics_bootstrap(
        y=y_test, y_pred=test_pred, ci_kwargs=dict(n_resamples=200)
    )
    # test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}
    # try:
    #     best_model.features = X_var # add feature names to model

    # except:
    #     print("can't assign feature names to model") 
    #     pass 
    try:
        if save_dir:
            Path(save_dir).parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(best_model, open(f"{save_dir}", "wb"))
    except:
        pass 
    return best_model, train_metrics, test_metrics, train_df, test_df, best_models


class EnsembleModel(object):
    def __init__(self, models, cov = None,coef_name=None, model_name_list=None):
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
        self.cov = cov

        self.model_name_list = (
            model_name_list
            if model_name_list
            else [f"model_{i}" for i in range(len(self.models))]
        )

        self.res = self._init_coeffeients_df()
        self._init_weights_dist()
    def reinit(self, new_coef=None, cov=None):
        self.__init__(self.models, coef_name = new_coef, cov=cov,model_name_list = self.model_name_list)

    def __repr__(self) -> str:
        return str(self.models)
    def _init_coeffeients_df(self):

        res = []
        for model_each in self.models:
            coef_ = model_each.coef_ if hasattr(model_each, "coef_") else model_each["model"].coef_
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
            res_df = pd.DataFrame(
                coef_,
                index = feature_names_in_
            )
            res_df.columns = ['coefficients']
            res_df.sort_values("coefficients", ascending=False)
            res.append(res_df)


        res = pd.concat(
            res,
            axis=1
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

    def predict(self, data,exclude=None, method="mean"):
        preds = []

        # check all feature in data 
        data = data.loc[:, self.features].copy()
        if self.cov:
            exclude = self.cov 

        for model in self.models:
            if exclude:  # 去除cov效应
                preds.append(get_predict(model, data,x_var=self.features, exclude = exclude))
            else:
                if hasattr(model, "predict_proba"):
                    preds.append(model.predict_proba(data)[:, 1])
                else:
                    preds.append(model.predict(data))

        if hasattr(self, "weight_model") and method == "weight_model":
            return self.weight_model.predict(np.array(preds).T)
        else:
            return np.mean(preds, axis=0)
    
    def fit_ensemble_weight(self, train_data, test_data, label="label",):
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
            test_data.loc[:, self.features + [label]].copy().dropna().reset_index(drop=True)
        )

        train_dict = {}
        test_dict = {}

        exclude = self.cov 


        for model_name, model in zip(self.model_name_list, self.models):
            if hasattr(model, "predict_proba"):
                train_dict[model_name] = model.predict_proba(train_data[self.features])[
                    :, 1
                ]
                test_dict[model_name] = model.predict_proba(test_data[self.features])[:, 1]

            else:
                if exclude:
                    train_dict[model_name] = get_predict(model, train_data[self.features], exclude = exclude)
                    test_dict[model_name] = get_predict(model, test_data[self.features], exclude = exclude)
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
    save_dir=None,
):

    if n_jobs == 1:
        random_stats = [i for i in np.random.randint(0, 100000, n_resample)]
        res = []
        for i in tqdm(random_stats):
            train_df_sample = train_df.sample(frac=1, replace=True, random_state=i)
            best_model, *_ = fit_best_model(
                train_df_sample, test_df, X_var, y_var, method_list, cv, verbose, save_dir=f"{save_dir}/{i}.pkl" if save_dir else None
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
                f"{save_dir}/{i}.pkl" if save_dir else None,
            )
            for i in tqdm(random_stats)
        )

    model = EnsembleModel(res, coef_name=X_var, model_name_list=None)

    train_df[f"{y_var}_pred"] = model.predict(train_df[X_var])
    test_df[f"{y_var}_pred"] = model.predict(test_df[X_var])

    to_cal_train = train_df[[y_var, f"{y_var}_pred"]].copy().dropna()
    train_metrics = cal_binary_metrics_bootstrap(
        y=to_cal_train[y_var].values,
        y_pred=to_cal_train[f"{y_var}_pred"].values,
        ci_kwargs=dict(n_resamples=200),
    )
    to_cal_test = test_df[[y_var, f"{y_var}_pred"]].copy().dropna()
    test_metrics = cal_binary_metrics_bootstrap(
        y=to_cal_test[y_var].values,
        y_pred=to_cal_test[f"{y_var}_pred"].values,
        ci_kwargs=dict(n_resamples=200),
    )
    test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}
    return model, train_metrics, test_metrics, train_df, test_df