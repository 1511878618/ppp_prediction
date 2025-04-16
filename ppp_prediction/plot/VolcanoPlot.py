import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
import matplotlib.colors as mcolors
def plot_vocano(
    data,
    x,
    y,
    pval,  # pvalue column name
    cutoff=0.05,  # cutoff for pvalue
    hue=None,
    cmap=None,
    vmin=None,
    vmax=None,
    center=None,
    ax=None,
):
    if cmap is None:
        cmap = mcolors.LinearSegmentedColormap.from_list(
            name="my",
            colors=[
                "#337bb7",
                "#4f99c7",
                "#a8d0e4",
                "#d1e6f1",
                "#fdd5c0",
                "#ef9c7b",
                "#de6e58",
                "#b2182e",
            ],
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    c_plt_data = data.sort_values(y, ascending=False).copy()
    c_plt_data["mlog10p"] = (-np.log10(c_plt_data[pval])).clip(0, 300)

    vmin = vmin if vmin is not None else np.quantile(c_plt_data[y], 0.1)
    vmax = vmax if vmax is not None else np.quantile(c_plt_data[y], 0.9)
    center = center if center is not None else 0
    hue = y if hue is None else hue

    sns.scatterplot(
        x=y,
        y="mlog10p",
        hue=y,
        palette=cmap,
        hue_norm=mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax),
        data=c_plt_data.query(f"{pval} < {cutoff}"),
        s=40,
        lw=0.25,
        ec="black",
        legend=False,
    )
    sns.scatterplot(
        x=y,
        y="mlog10p",
        color="#ebebeb",
        ec="#ebebeb",
        data=c_plt_data.query(f"{pval} > {cutoff}"),
        s=40,
        alpha=1,
        legend=False,
    )

    annotate_df = pd.concat(
        [
            # c_plt_data.sort_values("mlog10p", ascending=False).head(5),
            c_plt_data.query(f"{pval} < {cutoff} and {y} > {center}")
            .sort_values("coef", ascending=False)
            .head(5),
            # c_plt_data.sort_values("mlog10p", ascending=True).head(5),
            c_plt_data.query(f"{pval} < {cutoff} and {y} < {center}")
            .sort_values("coef", ascending=True)
            .head(5),
        ]
    ).drop_duplicates()

    texts = [
        ax.text(
            row[y],
            row["mlog10p"],
            row[x],
            ha="center",
            va="center",
            fontsize=8,
            color="black",
        )
        for i, row in annotate_df.iterrows()
    ]

    adjust_text(
        texts,  # expand text bounding boxes by 1.2 fold in x direction and 2 fold in y direction
        # ensure the labeling is clear by adding arrows
    )
    return ax

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
