import statsmodels.api as sm

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.stats import bootstrap


def calibration_curve_df(y_true, y_pred, k=10, n_resample=1000):
    plt_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})

    plt_df["y_pred_rank"] = plt_df["y_pred"].rank(
        pct=True, na_option="keep", ascending=True, method="average"
    )

    bins_start = np.linspace(0, 1, k + 1)
    bins_end = np.linspace(0, 1, k + 1)[1:]
    bins_labels = [f"{i:.0%}-{j:.0%}" for i, j in zip(bins_start, bins_end)]

    plt_df["y_pred_bins"] = pd.cut(
        plt_df["y_pred_rank"],
        bins=k,
        labels=bins_labels,
        include_lowest=True,
    )

    if not n_resample:
        plt_df_group = (
            plt_df.groupby("y_pred_bins")
            .apply(lambda x: pd.Series({"mean_y_true": x.y_true.mean()}))
            .reset_index(drop=False)
        )
    else:
        # 定义一个函数来计算均值
        def mean_bootstrap(data):
            # 使用bootstrap计算均值的置信区间
            res = bootstrap(data=(data,), statistic=np.mean, n_resamples=n_resample)

            return (
                np.mean(data),
                res.confidence_interval.low,
                res.confidence_interval.high,
            )

        # 对每个分位数进行bootstrap抽样
        plt_df_group = (
            plt_df.groupby("y_pred_bins")
            .apply(
                lambda x: pd.Series(
                    list(mean_bootstrap(x["y_true"])) + [x["y_pred"].mean()],
                    index=["mean_true", "ci_low", "ci_high", "mean_pred"],
                ).T
            )
            .reset_index(drop=False)
        )

    return plt_df_group


def _calibration_curve_plot(
    y_true, y_pred, k=10, n_resample=1000, offset=0, color="black", ax=None, label=None
):
    plt_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    _plt_data = calibration_curve_df(
        plt_df["y_true"], plt_df["y_pred"], k=k, n_resample=n_resample
    )
    x = np.arange(len(_plt_data["y_pred_bins"])) + offset
    ax.errorbar(
        x=x,
        y=_plt_data["mean_true"],
        yerr=[
            _plt_data["mean_true"] - _plt_data["ci_low"],
            _plt_data["ci_high"] - _plt_data["mean_true"],
        ],
        fmt="o",
        marker="o",
        mfc=color,
        color=color,
        capsize=4,
        label=label,
    )
    # y_lowess = sm.nonparametric.lowess(_plt_data["mean_true"], x, frac=0.5)
    # ax.plot(y_lowess[:, 0], y_lowess[:, 1], color=color, linestyle="--")
    # sns.lineplot

    ax.set_xticks(np.arange(len(_plt_data["y_pred_bins"])))
    ax.set_xticklabels(_plt_data["y_pred_bins"])
    return ax


def calibration_dot_plot(
    y_true,
    y_pred,
    data,
    hue=None,
    hue_order=None,
    ax=None,
    color=None,
    k=10,
    n_resample=1000,
    offset = 0.2,
):

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    if hue:
        hue_order = hue_order if hue_order is not None else data[hue].unique()
        offset_list = np.linspace(-offset, offset, len(hue_order))
        for idx in range(len(hue_order)):
            current_data = data[data[hue] == hue_order[idx]]
            color = (
                color
                if color is not None
                else sns.palettes.color_palette("Set1", len(hue_order))
            )

            _calibration_curve_plot(
                current_data[y_true],
                current_data[y_pred],
                k=k,
                n_resample=n_resample,
                offset=offset_list[idx],
                color=color[idx] if hasattr(color, "__iter__") else color,
                ax=ax,
                label=hue_order[idx],
            )

    else:
        _calibration_curve_plot(
            data[y_true],
            data[y_pred],
            k=k,
            n_resample=n_resample,
            offset=0,
            color=color,
            ax=ax,
            label=y_pred,
        )

    ax.legend(loc="upper left")

    ax.yaxis.grid(color="grey", linestyle="-", linewidth=1, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Risk Score Decile", fontsize=12)
    ax.set_ylabel("Observed Event Rate", fontsize=12)
    return ax
