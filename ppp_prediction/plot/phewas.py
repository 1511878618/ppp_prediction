from matplotlib import lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from adjustText import adjust_text
import pandas as pd
from pycirclize.track import Track
from pycirclize import utils
from matplotlib.projections.polar import PolarAxes


from pycirclize import Circos
import numpy as np
from matplotlib.lines import Line2D


def plot_phewas(
    data,
    p,
    coef,
    name,
    k=5,
    ax=None,
    hue=None,
    hue_order=None,
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
    data["log10p"] = data["log10p"].replace(np.inf, 365)

    data["marker"] = data[coef].apply(lambda x: "up" if x > 0 else "down")

    if hue is None:
        data.reset_index(drop=False, inplace=True)
        data["color_region"] = pd.cut(
            data["index"],
            bins=np.linspace(0, len(data["index"]), 10),
            include_lowest=True,
        )
        need_cols.append("color_region")
        hue = "color_region"
        del data["index"]
    # sort by hue
    if hue_order is not None:
        data[hue] = (
            data[hue].astype("category").cat.set_categories(hue_order, ordered=True)
        )
    data = (
        data.sort_values(hue, ascending=False)
        .reset_index(drop=True)
        .reset_index(drop=False)
    )
    # return data

    # top k
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
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
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


def Add_track_text(
    Track: Track,
    text: str,
    x: float | None = None,
    r: float | None = None,
    *,
    adjust_rotation: bool = True,
    orientation: str = "horizontal",
    ignore_range_error: bool = False,
    outer: bool = True,
    only_rotation: bool = False,
    **kwargs,
) -> None:
    """Plot text

    This func add outer and only_rotation params from Track.text functions; this supports for text rotation calculation.

    Parameters
    ----------
    text : str
        Text content
    x : float | None, optional
        X position. If None, track center x position is set.
    r : float | None, optional
        Radius position. If None, track center radius position is set.
    adjust_rotation : bool, optional
        If True, text rotation is auto set based on `x` and `orientation` params.
    orientation : str, optional
        Text orientation (`horizontal` or `vertical`)
        If adjust_rotation=True, orientation is used for rotation calculation.
    ignore_range_error : bool, optional
        If True, ignore x position range error
        (ErrorCase: `not track.start <= x <= track.end`)
    **kwargs : dict, optional
        Text properties (e.g. `size=12, color="red", va="center", ...`)
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html>
    """
    # If value is None, center position is set.
    x = Track.center if x is None else x
    r = Track.r_center if r is None else r

    rad = Track.x_to_rad(x, ignore_range_error)
    if adjust_rotation:
        params = utils.plot.get_label_params_by_rad(
            rad, orientation, only_rotation=only_rotation, outer=outer
        )
        kwargs.update(params)

    if "ha" not in kwargs and "horizontalalignment" not in kwargs:
        kwargs.update(dict(ha="center"))
    if "va" not in kwargs and "verticalalignment" not in kwargs:
        kwargs.update(dict(va="center"))

    def plot_text(ax: PolarAxes) -> None:
        ax.text(rad, r, text, **kwargs)

    Track._plot_funcs.append(plot_text)


def plot_phewas_circle(
    data,
    p,
    coef,
    hue,
    name,
    hue_order=None,
    gap=1,
    title=None,
    cutoff=None,
    scatter_marker_hue=None,  # not work
    scatter_marker_hue_marker=None,  # not work
    ax=None,
    figsize=(10, 10),
    cmap="RdBu_r",
    hue_min=None,
    hue_max=None,
):
    sectors = data.groupby(hue).apply(lambda x: x.shape[0]).to_dict()
    np.random.seed(0)
    plt_data = data
    total = plt_data.shape[0]

    tgt = title
    group_in_name = name
    groupby = hue

    if hue_order is None:
        hue_order = plt_data[hue].unique()
    group_order = hue_order

    plt_data["LOG10P"] = -np.log10(plt_data[p])
    plt_data["LOG10P"] = plt_data["LOG10P"].replace(np.inf, 365)
    scatter_y = "LOG10P"
    if cutoff is None:
        cutoff = -np.log10(0.05 / total)
    else:
        cutoff = -np.log10(cutoff)
    y_cutoff = cutoff

    scatter_hue = coef

    plt_data[groupby] = (
        plt_data[groupby]
        .astype("category")
        .cat.set_categories(group_order, ordered=True)
    )

    sectors = (
        plt_data.groupby(groupby)
        .apply(lambda x: int(365 * max(x.shape[0] - gap, 1) / total))
        .to_dict()
    )

    vmin = plt_data[scatter_y].min()
    vmax = max(plt_data[scatter_y].max(), y_cutoff)

    if scatter_hue:
        hue_min = plt_data[scatter_hue].min() if hue_min is None else hue_min
        hue_max = plt_data[scatter_hue].max() if hue_max is None else hue_max

    print(f"vmin: {vmin}, vmax: {vmax}")

    track_r = [100, 80, 60]
    track_fontsize = [10, 7]

    circos = Circos(sectors, space=4)
    circos.text(f"{tgt}", size=15)
    circos.line(r=40, color="grey", lw=2, alpha=0.8)
    circos_sectors = circos.sectors
    num_of_sectors = len(circos_sectors)
    for idx, sector in enumerate(circos_sectors):
        track_data = plt_data[plt_data[groupby] == sector.name].sort_values(coef)

        # laber track
        label_track = sector.add_track((track_r[0], track_r[0]))
        Add_track_text(
            label_track,
            sector.name,
            x=sector.center,
            # text = "123",
            r=track_r[0] + 1,
            color="black",
            fontsize=track_fontsize[0],
            # ha='left',
            # va='bottom',
            orientation="vertical",
            adjust_rotation=True,
            fontweight="bold",
        )
        label_track.axis(ec="grey")

        # each subgroup
        subgroup_data = track_data[group_in_name].tolist()
        num_group_in_name = len(subgroup_data)
        group_in_name_track = sector.add_track((track_r[1], track_r[1]))
        group_in_name_track.axis(ec="grey")
        group_in_name_track_x = np.linspace(
            0, group_in_name_track.size, num_group_in_name
        )
        for i, group in enumerate(subgroup_data):
            Add_track_text(
                group_in_name_track,
                group,
                x=group_in_name_track_x[i],
                r=track_r[1],
                color="black",
                fontsize=track_fontsize[1],
                fontweight="bold",
                orientation="vertical",
                adjust_rotation=True,
            )

        # pvalue scatter track
        scatter_track = sector.add_track((50, 70))

        x = np.linspace(0, group_in_name_track.size, num_group_in_name)
        y = track_data[scatter_y].values

        scatter_df_dict = {
            "x": x,
            "y": y,
        }
        if scatter_marker_hue:

            scatter_df_dict["marker_hue"] = track_data[scatter_marker_hue].values

        if scatter_hue:
            scatter_df_dict["scatter_hue"] = track_data[scatter_hue].values

        scatter_df = pd.DataFrame(scatter_df_dict)

        if scatter_marker_hue:

            for scatter_marker_hue_name, scatter_marker_hue_df in scatter_df.groupby(
                "marker_hue"
            ):

                scatter_makrer_hue_marker = scatter_marker_hue_marker[
                    scatter_marker_hue_name
                ]
                scatter_track.scatter(
                    x=scatter_marker_hue_df["x"].values,
                    y=scatter_marker_hue_df["y"].values,
                    vmin=vmin,
                    vmax=vmax,
                    s=20,
                    cmap="RdBu_r",
                    c=(
                        scatter_marker_hue_df["scatter_hue"].values
                        if scatter_hue is not None
                        else None
                    ),
                    norm=plt.Normalize(vmin, vmax),
                    label=scatter_marker_hue_name,
                    marker=scatter_makrer_hue_marker,
                )

        else:
            scatter_track.scatter(
                x=scatter_df["x"].values,
                y=scatter_df["y"].values,
                vmin=vmin,
                vmax=vmax,
                s=20,
                cmap=cmap,
                c=scatter_df["scatter_hue"].values if scatter_hue is not None else None,
                norm=plt.Normalize(hue_min, hue_max),
            )

        scatter_track.axis(visible=False)

        if y_cutoff:
            median = scatter_track._y_to_r(y_cutoff, vmin, vmax)
            circos.line(
                r=median,
                color="red",
                linestyle="-",
                lw=0.5,
                alpha=0.5,
            )
        # scatter_track.line(x=[0, group_in_name_track.size], y=[0, 0], color="grey", linestyle="-", vmin=vmin, vmax=vmax, lw=2, alpha=.5)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    if scatter_marker_hue:
        ## plot legend
        scatter_handels = [
            Line2D(
                [],
                [],
                color="orange",
                marker=scatter_makrer_hue_marker,
                label=scatter_marker_hue_name,
                ms=6,
                ls="None",
            )
            for scatter_marker_hue_name, scatter_makrer_hue_marker in scatter_marker_hue_marker.items()
        ]

        scatter_legend = ax.legend(
            handles=scatter_handels,
            bbox_to_anchor=(0.93, 0.98),
            fontsize=8,
            title="Significant",
            handlelength=2,
        )

    circos.colorbar(
        # bounds=(0.93, 0.98, 0.4, 0.03),
        bounds=(0.96, 0.98, 0.02, 0.3),
        vmin=hue_min,
        vmax=hue_max,
        cmap=cmap,
        # orientation="horizontal",
        colorbar_kws=dict(label=coef),
        # tick_kws=dict(labelsize=12, colors="red"),
    )
    fig = circos.plotfig(ax=ax)

    return ax
