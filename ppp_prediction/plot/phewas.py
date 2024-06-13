from matplotlib import lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from adjustText import adjust_text
import pandas as pd


def plot_pheweb(
    data,
    p,
    coef,
    name=None,
    k=5,
    ax=None,
    hue=None,
    pallete=None,
    cutoff=None,
    xlab=None,
    ylab=None,
    scatter_kwargs=None,
    hline_kwargs=None,
    text_kwargs=None,
    legend_kwargs=None,
    legend_marker_color="black",  # C74546"",
):
    need_cols = [p, coef]
    if name is not None:
        need_cols.append(name)
    if hue is not None:
        need_cols.append(hue)

    data = data[need_cols].copy()
    data["log10p"] = -np.log10(data[p])
    data["marker"] = data[coef].apply(lambda x: "up" if x > 0 else "down")
    data.reset_index(inplace=True)

    if hue is None:
        data["color_region"] = pd.cut(
            data["index"],
            bins=np.linspace(0, len(data["index"]), 10),
            include_lowest=True,
        )
        need_cols.append("color_region")
        hue = "color_region"

    topk = data.sort_values("log10p", ascending=False).head(k)

    if cutoff is None:
        cutoff = 1e-2 / len(data)

    scatter_kwargs_used = dict(
        s=60,
        edgecolor="black",
        alpha=0.8,
        style="marker",
        markers={"up": "^", "down": "v"},
    )
    if scatter_kwargs is not None:
        scatter_kwargs_used.update(scatter_kwargs)

    if ax is None:
        fig, ax = plt.subplots()

    # plot
    sns.scatterplot(
        data=data,
        x="index",
        y="log10p",
        hue=hue,
        palette=pallete,
        ax=ax,
        **scatter_kwargs_used,
    )

    # hline
    hline_kwargs_used = dict(color="grey", linestyle="--", alpha=0.5)
    if hline_kwargs is not None:
        hline_kwargs_used.update(hline_kwargs)

    ax.axhline(-np.log10(cutoff), **hline_kwargs_used)

    # text
    text_kwargs_used = dict(fontsize=12, color="black", alpha=0.8)
    if text_kwargs is not None:
        text_kwargs_used.update(text_kwargs)

    texts = [
        ax.text(row["index"], row["log10p"], row[name], **text_kwargs_used)
        for i, row in topk.iterrows()
    ]
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color="black", lw=0.5), ax=ax)

    # set xlab
    if xlab is None:
        xlab = name
    if ylab is None:
        ylab = "-Log10(p-value)"
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # set legend
    handles = [
        mlines.Line2D(
            [],
            [],
            color=legend_marker_color,
            markerfacecolor="white",
            marker="^",
            linestyle="None",
            markersize=10,
            label="Up",
        ),
        mlines.Line2D(
            [],
            [],
            color=legend_marker_color,
            marker="v",
            linestyle="None",
            markerfacecolor="white",
            markersize=10,
            label="Down",
        ),
    ]

    # legend
    legend_kwargs_used = dict(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=2,
        frameon=True,
        prop={"size": 14},
        markerscale=1.5,
    )
    if legend_kwargs is not None:
        legend_kwargs_used.update(legend_kwargs)
    ax.legend(
        handles=handles,
        **legend_kwargs_used,
    )
    # set tick
    ax.tick_params(axis="y", direction="in", length=5)
    if hue == "color_region":
        ax.set_xticks([])
    else:
        # print("may have problems now ")
        xticks = np.ceil(data.groupby(hue)["index"].mean())
        ax.set_xticks(xticks)
        ax.set_xticklabels(data.groupby(hue)[hue].first())
    return ax
