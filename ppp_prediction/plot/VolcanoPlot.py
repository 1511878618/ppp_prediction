import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
def VolcanoPlot(
    data,
    x,
    pval,
    var,
    agg_on,
    cutoff=1e-2,
    annot_k=10,
    adjust_pval_method="fdr_bh",
    pallete="Set2",
    ax=None,
    scatter_kwargs=None,
    line_kwargs=None,
):
    # prepare data
    data = data[[x, pval, var, agg_on]].copy()

    if adjust_pval_method:
        data = generate_multipletests_result(data, pval, method=adjust_pval_method)
    data["-log10(p)"] = -np.log10(data[pval])

    def agg_Func(df):
        # agg
        df.sort_values("-log10(p)", inplace=True, ascending=False)
        top_series = df.iloc[0]
        size = df[df[pval] < cutoff].shape[0]
        top_series["size"] = size
        return top_series

    data = data.groupby(var).apply(agg_Func).reset_index(drop=True)

    data["is_sig"] = data[pval] < cutoff

    # plot
    if ax is None:
        fig, ax = plt.subplots()

    scatter_kwargs_used = dict(
        s=50,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
        sizes=(50, 200),
        size="size",
    )

    if scatter_kwargs is not None:
        if "palette" in scatter_kwargs:
            scatter_kwargs.pop("pallet", None)
            print(
                f"palette is not supported in scatter_kwargs, just pass it to the function"
            )
        scatter_kwargs_used.update(scatter_kwargs)
    norm_size = data["size"][0]
    sns.scatterplot(
        data=data.query("is_sig == True"),
        x=x,
        y="-log10(p)",
        hue=agg_on,
        palette=pallete,
        ax=ax,
        **scatter_kwargs_used,
    )
    ax.legend(
        loc="upper left", bbox_to_anchor=(1, 1), fontsize=10, ncol=1, prop={"size": 15}
    )

    # plot non-significant
    ax.scatter(
        data.query("is_sig == False")[x],
        data.query("is_sig == False")["-log10(p)"],
        color="gray",
        s=norm_size,
        alpha=0.4,
    )

    line_kwargs_used = dict(color="red", linestyle="-")
    if line_kwargs is not None:
        line_kwargs_used.update(line_kwargs)
    ax.axhline(-np.log10(cutoff), **line_kwargs_used)

    if annot_k:
        topk = (
            data.query("is_sig == True")
            .sort_values("-log10(p)", ascending=False)
            .head(annot_k)
        )
        texts = [
            ax.text(
                row[x],
                row["-log10(p)"],
                row[var],
                fontsize=10,
                verticalalignment="center",
                horizontalalignment="center",
                bbox=dict(
                    facecolor="none", edgecolor="black", boxstyle="round,pad=0.5"
                ),
            )
            for i, row in topk.iterrows()
        ]
        adjust_text(texts)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax
