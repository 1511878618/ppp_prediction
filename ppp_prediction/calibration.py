# from sklearn.calibration import (
#     CalibratedClassifierCV,
#     CalibrationDisplay,
#     calibration_curve,
# )
# import seaborn as sns
# from sklearn.isotonic import IsotonicRegression
# from sklearn.linear_model import LogisticRegression


import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB

import numpy as np

from sklearn.svm import LinearSVC

from collections import defaultdict

import pandas as pd

from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class NaivelyCalibratedLinearSVC(LinearSVC):
    """LinearSVC with `predict_proba` method that naively scales
    `decision_function` output for binary classification."""

    def fit(self, X, y):
        super().fit(X, y)
        df = self.decision_function(X)
        self.df_min_ = df.min()
        self.df_max_ = df.max()

    def predict_proba(self, X):
        """Min-max scale output of `decision_function` to [0, 1]."""
        df = self.decision_function(X)
        calibrated_df = (df - self.df_min_) / (self.df_max_ - self.df_min_)
        proba_pos_class = np.clip(calibrated_df, 0, 1)
        proba_neg_class = 1 - proba_pos_class
        proba = np.c_[proba_neg_class, proba_pos_class]
        return proba


class NaivelyCalibratedLinearSVC(LinearSVC):
    """LinearSVC with `predict_proba` method that naively scales
    `decision_function` output."""

    def fit(self, X, y):
        super().fit(X, y)
        df = self.decision_function(X)
        self.df_min_ = df.min()
        self.df_max_ = df.max()

    def predict_proba(self, X):
        """Min-max scale output of `decision_function` to [0,1]."""
        df = self.decision_function(X)
        calibrated_df = (df - self.df_min_) / (self.df_max_ - self.df_min_)
        proba_pos_class = np.clip(calibrated_df, 0, 1)
        proba_neg_class = 1 - proba_pos_class
        proba = np.c_[proba_neg_class, proba_pos_class]
        return proba


def get_calibration_df(
    data,
    obs,
    pred,
    followup=None,
    group=None,
    n_bins=10,
    # strategy="quantile",
):
    data = data.copy()

    if followup is None:
        followup = "followup"
        data[followup] = 1

    # if strategy == "quantile":  # Determine bin edges by distribution of data
    #     quantiles = np.linspace(0, 1, n_bins + 1)
    #     bins = np.percentile(y_prob, quantiles * 100)
    # elif strategy == "uniform":
    #     bins = np.linspace(0.0, 1.0, n_bins + 1)
    # else:
    #     raise ValueError(
    #         "Invalid entry to 'strategy' input. Strategy "
    #         "must be either 'quantile' or 'uniform'."
    #     )

    if group is not None:

        data = data.groupby(group).apply(
            lambda x: x.assign(
                decile=pd.qcut(x[pred], n_bins, labels=False, duplicates="drop")
            )
        )
        data = (
            data.groupby([group, "decile"])
            .apply(
                lambda x: pd.Series(
                    {
                        "obsRate": (x[obs] / x[followup]).mean(),
                        "obsRate_SE": (x[obs] / x[followup]).std() / np.sqrt(len(x)),
                        "obsNo": x[obs].sum(),
                        "predMean": x[pred].mean(),
                    }
                )
            )
            .reset_index()
        )
    else:
        data = data.assign(
            decile=pd.qcut(data[pred], n_bins, labels=False, duplicates="drop")
        )
        data = (
            data.groupby("decile")
            .apply(
                lambda x: pd.Series(
                    {
                        "obsRate": (x[obs] / x[followup]).mean(),
                        "obsRate_SE": (x[obs] / x[followup]).std() / np.sqrt(len(x)),
                        "obsNo": x[obs].sum(),
                        "predMean": x[pred].mean(),
                    }
                )
            )
            .reset_index()
        )
    data["obsRate_UCI"] = np.clip(
        data["obsRate"] + 1.96 * data["obsRate_SE"], a_max=1, a_min=None
    )
    data["obsRate_LCI"] = np.clip(
        data["obsRate"] - 1.96 * data["obsRate_SE"], a_min=0, a_max=None
    )
    return data


def calibrate(X_train, y_train, X_test=None, y_test=None, n_bins=5, need_scale=True):

    #
    lr = LogisticRegressionCV(
        Cs=np.logspace(-6, 6, 101), cv=10, scoring="neg_log_loss", max_iter=1_000
    )
    rfc = RandomForestClassifier(random_state=42)

    gnb = GaussianNB()
    gnb_isotonic = CalibratedClassifierCV(gnb, cv=10, method="isotonic")
    gnb_sigmoid = CalibratedClassifierCV(gnb, cv=10, method="sigmoid")

    svc = NaivelyCalibratedLinearSVC(C=1.0)
    svc_isotonic = CalibratedClassifierCV(svc, cv=10, method="isotonic")
    svc_sigmoid = CalibratedClassifierCV(svc, cv=10, method="sigmoid")

    clf_list = [
        (lr, "Logistic Regression"),
        # (rfc, "Random forest"),
        (gnb, "Naive Bayes"),
        (gnb_isotonic, "Naive Bayes + Isotonic"),
        (gnb_sigmoid, "Naive Bayes + Sigmoid"),
        # (svc, "SVC"),
        # (svc_isotonic, "SVC + Isotonic"),
        # (svc_sigmoid, "SVC + Sigmoid"),
    ]
    if need_scale:
        clf_list = [
            (Pipeline([("scaler", StandardScaler()), ("model", model)]), name)
            for model, name in clf_list
        ]

    if X_test is None and y_test is None:
        X_test = X_train
        y_test = y_train
    elif X_test is not None and y_test is not None:
        pass
    else:
        raise ValueError("X_test and y_test should be both None or not None")

    scores = defaultdict(list)
    clf_calibration_df_list = []
    clf_dict = {}
    for i, (clf, name) in enumerate(clf_list):
        # fit model
        clf.fit(X_train, y_train)

        # cal metrics
        y_prob = clf.predict_proba(X_test)
        y_pred = clf.predict(X_test)

        clf_calibration_df = get_calibration_df(
            data=pd.DataFrame({"obs": y_test, "pred": y_prob[:, 1]}),
            obs="obs",
            pred="pred",
            followup=None,
            group=None,
            n_bins=n_bins,
        )

        scores["Classifier"].append(name)
        for metric in [brier_score_loss, log_loss, roc_auc_score]:
            score_name = (
                metric.__name__.replace("_", " ").replace("score", "").capitalize()
            )
            scores[score_name].append(metric(y_test, y_prob[:, 1]))

        for metric in [precision_score, recall_score, f1_score]:
            score_name = (
                metric.__name__.replace("_", " ").replace("score", "").capitalize()
            )
            scores[score_name].append(metric(y_test, y_pred))

        clf_calibration_df_list.append(clf_calibration_df.assign(Classifier=name))
        clf_dict[name] = clf

    score_df = pd.DataFrame(scores).set_index("Classifier")
    score_df = score_df.sort_values(by="Brier  loss")
    score_df.round(decimals=3)

    clf_calibration_df_list_df = pd.concat(clf_calibration_df_list)
    return {
        "best_clf": clf_dict[score_df.index[0]],
        "best_clf_name": score_df.index[0],
        "score_df": score_df,
        "calibration_df": clf_calibration_df_list_df,
        "clf_dict": clf_dict,
    }


def plot_calibrate(
    calibration_df,
    x="predMean",
    y="obsRate",
    hue=None,
    ax=None,
    palette="tab20",
    ci=False,
    **kwargs,
):

    if ax is None:
        fig, ax = plt.subplots()

    calibration_df = calibration_df.copy()
    plot_boundary = max(calibration_df[x].max(), calibration_df[y].max())

    if ci:
        if f"{y}_LCI" not in calibration_df.columns:
            raise ValueError("CI column not found")
        if f"{y}_UCI" not in calibration_df.columns:
            raise ValueError("CI column not found")
        plot_boundary = max(
            plot_boundary,
            calibration_df[f"{y}_LCI"].max(),
            calibration_df[f"{y}_UCI"].max(),
        )

    plot_boundary = min(plot_boundary + plot_boundary * 0.1, 1)

    if hue is None:
        hue = "NewHue"
        calibration_df[hue] = "Calibration Line"

    if palette is None:
        palette = sns.color_palette(n_colors=calibration_df[hue].nunique())
    elif isinstance(palette, str):
        palette = sns.color_palette(palette, n_colors=calibration_df[hue].nunique())

    if isinstance(palette, list):
        palette = {
            name: color for name, color in zip(calibration_df[hue].unique(), palette)
        }

    line_kwargs = {"marker": "s", "linestyle": "-"}

    line_kwargs.update(**kwargs)

    ref_line_label = "Perfectly calibrated"

    ax.plot([0, 1], [0, 1], "k:", label=ref_line_label, color="gray", linestyle="--")

    for name, group in calibration_df.groupby(hue):

        if ci:
            ax.errorbar(
                group[x],
                group[y],
                yerr=[
                    group["obsRate"] - group["obsRate_LCI"],
                    group["obsRate_UCI"] - group["obsRate"],
                ],
                fmt="o",
                color=palette[name],
            )
        else:
            ax.plot(
                group[x],
                group[y],
                label=name,
                color=palette[name],
                **line_kwargs,
            )

    ax.set_xlim(0, plot_boundary)
    ax.set_ylim(0, plot_boundary)
    ax.legend(loc="lower right")
    xlabel = "Mean predicted probability"
    ylabel = "Observed event rate"
    ax.set(xlabel=xlabel, ylabel=ylabel)
    return ax


# import pandas as pd 
# def calibration_score(
#     raw_train_pred, raw_test_pred, train_y, method="logitstic", **kwargs
# ):
#     if method == "isotonic":
#         ir = IsotonicRegression(out_of_bounds="clip")
#         ir.fit(raw_train_pred, train_y)

#         pred_train_calibrated = ir.predict(raw_train_pred)
#         pred_test_calibrated = ir.predict(raw_test_pred)
#     elif method == "logitstic":
#         lr = LogisticRegression(
#             # class_weight="balanced",
#             max_iter=5000,
#             random_state=123,
#             **kwargs,
#         )
#         raw_train_pred = (
#             raw_train_pred.values
#             if isinstance(raw_train_pred, pd.Series)
#             else raw_train_pred
#         )
#         raw_test_pred = (
#             raw_test_pred.values
#             if isinstance(raw_test_pred, pd.Series)
#             else raw_test_pred
#         )
#         lr.fit(raw_train_pred.reshape(-1, 1), train_y)
#         pred_train_calibrated = lr.predict_proba(raw_train_pred.reshape(-1, 1))[:, 1]
#         pred_test_calibrated = lr.predict_proba(raw_test_pred.reshape(-1, 1))[:, 1]
#     else:
#         raise ValueError("method should be isotonic or logitstic")

#     return pred_train_calibrated, pred_test_calibrated


# def plot_calibration_score(
#     data,
#     fraction_of_positives="fraction_of_positives",
#     mean_predicted_value="mean_predicted_value",
#     ax=None,
#     hue=None,
#     **kwargs,
# ):
#     if ax is None:
#         fig, ax = plt.subplots()
#     up_bd = max([max(data[fraction_of_positives]), max(data[mean_predicted_value])])

#     sns.lineplot(
#         data=data,
#         x=mean_predicted_value,
#         y=fraction_of_positives,
#         ax=ax,
#         hue=hue,
#         **kwargs,
#     )

#     ax.set(
#         title="Calibration plot",
#         xlabel="Mean predicted value",
#         ylabel="Fraction of positives",
#     )

#     ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
#     ax.set_xlim(0, up_bd)
#     ax.set_ylim(0, up_bd)
#     ax.legend(bbox_to_anchor=(0.5, 1.05), loc="lower center")
#     return ax
