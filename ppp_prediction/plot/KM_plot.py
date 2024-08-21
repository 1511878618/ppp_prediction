from lifelines import CoxPHFitter
from collections import defaultdict

from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from ppp_prediction.cox import run_cox
from lifelines import CoxPHFitter
from collections import defaultdict

from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from plottable import ColumnDefinition, Table
import re
import forestplot as fp
from lifelines.plotting import add_at_risk_counts

from ppp_prediction.utils import generate_qcut_labels


def plot_km(
    plt_data,
    var,
    E,
    T,
    show_HR=False,
    cox_config=None,  # params to pass to run_cox
    cox_table_config=None,
    k=2,  # if K = None, var should be cat var
    ax=None,
    palette=None,
    test=True,
    bin_labels="auto",
    survTable=False,
    cumulative=True,
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    if palette is None:
        palette = sns.palettes.color_palette("Set1")
    else:
        if isinstance(palette, str):
            palette = sns.palettes.color_palette(palette, n_colors=3)
        elif not hasattr(palette, "__iter__"):
            raise ValueError("palette should be a iterable or string")
    if show_HR:
        if cox_config is None:
            cox_config = {}

        cov = cox_config.get("cov", [])
        cat_cov = cox_config.get("cat_cov", [])
        plt_data = plt_data[[var, E, T, *cov, *cat_cov]].copy()
    else:
        plt_data = plt_data[[var, E, T]].copy()
    if k:
        if bin_labels == "auto":
            bin_labels = generate_qcut_labels(k)

        elif bin_labels:
            assert len(bin_labels) == k
        else:
            bin_labels = None

        plt_data[f"{var}_bin"] = pd.qcut(
            plt_data[var], q=k, labels=bin_labels, duplicates="drop"
        )
    else:
        if len(plt_data[var].unique()) > 10:
            raise ValueError("Too many unique values for categorical variable")

        plt_data[f"{var}_bin"] = plt_data[var]
    if show_HR:

        res = run_cox(
            plt_data,
            var=f"{var}_bin",
            E=E,
            T=T,
            **cox_config,
            cat_var=f"{var}_bin",
        )
        # print(res)
        res = res[["var", "HR (95% CI)", "pvalue"]]
        # res["var"] = res["var"].apply(
        #     lambda x: re.findall(r"\[([^\]]+)\]", x)[0].lstrip("T.")
        # )
        res["pvalue"] = res["pvalue"].apply(lambda x: f"{x:.2e}")
        res.set_index("var", inplace=True)
        # add right bottom table
        # sub_ax = fig.add_axes([0.2, 0.6, 0.25, 0.25])
        # sub_ax = ax.inset_axes([0.5, 0.5, 0.4, 0.4])

        cox_config = {
            "sub_ax": [0.5, 0, 0.5, 0.3],
            "textprops": {"fontsize": 6},
        }
        # cox_table_config = cox_table_config or {}
        if cox_table_config:
            cox_config.update(cox_table_config)

        sub_ax = ax.inset_axes(cox_config.pop("sub_ax"))

        t = Table(res, ax=sub_ax, **cox_config)

    texts = ""
    kmf_list = []
    for idx, (name, grouped_df) in enumerate(plt_data.groupby(f"{var}_bin")):
        if grouped_df.shape[0] == 0:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(
            grouped_df[T],
            event_observed=grouped_df[E],
            label=name,
        )
        if cumulative:

            kmf.plot_cumulative_density(
                ax=ax,
                color=palette[idx],
                **kwargs,
            )
        else:
            kmf.plot_survival_function(
                ax=ax,
                color=palette[idx],
                **kwargs,
            )
        kmf_list.append(kmf)
        # texts += f"{name} have {E} {grouped_df[grouped_df[E] == 1].shape[0]}\n"
    if survTable:
        add_at_risk_counts(*kmf_list, ax=ax)

    if test:
        results = multivariate_logrank_test(
            plt_data[T],
            plt_data[f"{var}_bin"],
            plt_data[E],
        )
        N = plt_data[E].dropna().shape[0]
        msg = f"N = {N:,d}, Log-Rank, Chi-squared: {results.test_statistic:.2f},\n"

        if results.p_value < 0.001:
            msg += f" p-value: <0.001"
        else:
            msg += f" p-value: {results.p_value:.3f}"
        ax.text(
            0.5,
            1.0,
            msg,
            fontsize=12,
            ha="center",
            va="top",
            transform=ax.transAxes,
        )

    ax.set_title(f"{var} vs {T} and {E} at each bin", fontsize=14)
    ax.legend(title=f"{var}", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Survival time", fontsize=12)
    ax.set_ylabel(f"{E} Cumulative Density", fontsize=12)

    # ax.text(
    #     0.99,
    #     0.99,
    #     texts,
    #     fontsize=12,
    #     ha="right",
    #     va="top",
    #     transform=ax.transAxes,
    #     multialignment="left",
    # )
    return ax


def plot_KM_QT_k_percentile(
    plt_data,
    var,
    E,
    T,
    cov=None,  # for HR
    cat_cov=None,
    show_HR=False,
    k=2,
    ax=None,
    palette=None,
    test=True,
    bin_labels=None,
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    if palette is None:
        palette = sns.palettes.color_palette("Set1", n_colors=3)[::-1]
    else:
        if isinstance(palette, str):
            palette = sns.palettes.color_palette(palette, n_colors=3)
        elif not hasattr(palette, "__iter__"):
            raise ValueError("palette should be a iterable or string")

    plt_data = plt_data[[var, E, T]].copy()
    plt_data[var].rank(pct=True, ascending=True)
    if bin_labels:
        assert len(bin_labels) == k

    # plt_data[f"{var}_bin"] = pd.cut(
    #     plt_data[var].rank(pct=True, ascending=True),
    #     bins=k,
    #     labels=bin_labels,
    # )
    plt_data[f"{var}_bin"] = pd.qcut(
        plt_data[var], q=k, labels=bin_labels, duplicates="drop"
    )
    if show_HR:
        res = run_cox(
            plt_data, var=var, E=E, T=T, cov=cov, cat_cov=cat_cov, cat_var=var
        )

    texts = ""
    for idx, (name, grouped_df) in enumerate(plt_data.groupby(f"{var}_bin")):
        kmf = KaplanMeierFitter()
        kmf.fit(
            grouped_df[T],
            event_observed=grouped_df[E],
            label=name,
        )

        kmf.plot_cumulative_density(
            ax=ax,
            color=palette[idx],
            **kwargs,
        )
        # texts += f"{name} have {E} {grouped_df[grouped_df[E] == 1].shape[0]}\n"
    if test:
        results = multivariate_logrank_test(
            plt_data[T],
            plt_data[f"{var}_bin"],
            plt_data[E],
        )
        N = plt_data[E].dropna().shape[0]
        msg = f"N = {N:,d}, Log-Rank, Chi-squared: {results.test_statistic:.2f},\n"
        if results.p_value < 0.001:
            msg += f" p-value: <0.001"
        else:
            msg += f" p-value: {results.p_value:.3f}"
        ax.text(
            0.5,
            1.0,
            msg,
            fontsize=12,
            ha="center",
            va="top",
            transform=ax.transAxes,
        )

    ax.set_title(f"{var} vs {T} and {E} at each bin", fontsize=8)
    ax.legend(title=f"{var}_percentile", fontsize=8)

    # ax.text(
    #     0.99,
    #     0.99,
    #     texts,
    #     fontsize=12,
    #     ha="right",
    #     va="top",
    #     transform=ax.transAxes,
    #     multialignment="left",
    # )
    return ax


def plot_KM_percentile(plt_data, var, E, T, ax=None, palette=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    if palette is None:
        palette = sns.palettes.color_palette("Set1", n_colors=3)[::-1]
    else:
        if isinstance(palette, str):
            palette = sns.palettes.color_palette(palette, n_colors=3)
        elif not hasattr(palette, "__iter__"):
            raise ValueError("palette should be a iterable or string")
    k = 10
    plt_data = plt_data[[var, E, T]].copy()
    plt_data[var].rank(pct=True, ascending=True)
    plt_data[f"{var}_bin"] = pd.cut(
        plt_data[var].rank(pct=True, ascending=True), bins=k, ordered=True
    )

    bins = plt_data[f"{var}_bin"].cat.categories

    to_plot_bin = [bins[0], bins[5], bins[-1]]
    to_plot_bin_name = ["Bottom 10%", "Middle 10%", "Top 10%"]

    for idx, name in enumerate(to_plot_bin):
        # print(f"{var}_bin == @name")

        # grouped_df = plt_data.query(f"{var}_bin == @name")
        grouped_df = plt_data[plt_data[f"{var}_bin"] == name]
        # print(grouped_df)
        kmf = KaplanMeierFitter()
        kmf.fit(
            grouped_df[T], event_observed=grouped_df[E], label=to_plot_bin_name[idx]
        )

        kmf.plot_cumulative_density(ax=ax, color=palette[idx], **kwargs)

    ax.set_title(f"{var} vs {T} and {E} at each bin", fontsize=8)
    ax.legend(title=f"{var}_percentile", fontsize=8)

    # ax.text(
    #     0.99,
    #     0.99,
    #     texts,
    #     fontsize=12,
    #     ha="right",
    #     va="top",
    #     transform=ax.transAxes,
    #     multialignment="left",
    # )
    return ax
