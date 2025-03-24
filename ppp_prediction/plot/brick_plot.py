import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from adjustText import adjust_text
import pandas as pd


def set_bottom(data, y, center=0):

    pos_data = data[data[y] > center].reset_index(drop=True)
    neg_data = data[data[y] < center].reset_index(drop=True)
    non_data = data[data[y] == center].reset_index(drop=True)

    pos_bottom = 0
    if pos_data.shape[0] != 0:
        for idx, row in pos_data.iterrows():
            pos_data.loc[idx, "bottom"] = pos_bottom
            pos_bottom += row[y]
    neg_bottom = 0
    if neg_data.shape[0] != 0:
        for idx, row in neg_data.iterrows():

            neg_data.loc[idx, "bottom"] = neg_bottom
            neg_bottom += row[y]
    return pd.concat([pos_data, non_data, neg_data])


def brick_plot(
    data,
    direction,
    x,
    hue=None,
    hue_order=None,
    order=None,
    cmap="Set3",
    center=0,
    ax=None,
    annotate_kwargs=None,
    **kwargs,
):
    """
        Parameters:
        -----------
        data : pandas.DataFrame
            The input data for plotting.
        direction : str
            The column name in `data` representing the direction values.
        x : str
            The column name in `data` representing the categorical variable.
        hue : str, optional
            The column name in `data` representing the hue variable. Default is None.
        hue_order : list, optional
            The order of hue categories. Default is None.
        order : list, optional
            The order of categories in `x`. Default is None.
        cmap : str, list, or dict, optional
            The colormap to use for the plot. Default is "Set3".
        center : int or float, optional
            The center value for the direction variable. Default is 0.
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot. Default is None.
        annotate_kwargs : dict, optional
            Additional keyword arguments for annotations. Default is None.
        **kwargs : additional keyword arguments
            Additional arguments to pass to the bar plot.
        Returns:
        --------
        ax : matplotlib.axes.Axes
            The axes with the brick plot.
        --------
        Example:
        import numpy as np
        from ppp_prediction.plot.brick_plot import brick_plot
    
        demo_data = pd.DataFrame(
            {
                "var": np.random.choice([f"A_{i}" for i in range(10)], 100),
                "direction": np.random.choice([1, 0, -1], 100),
            }
        )

        brick_plot(demo_data, direction="direction", x="var", hue="var")

    """

    data = data.copy()
    if ax is None:
        fig, ax = plt.subplots()
    if order is None:
        direction_sum = data.groupby(x)[direction].sum().sort_values(ascending=False)
        order = direction_sum.index.tolist()
    data[x] = pd.Categorical(data[x], order, ordered=True)

    if hue is None:
        if hue_order is not None:
            raise ValueError("hue_order is not None while hue is None")
        hue = "hue"
        data[hue] = "default"

    if hue_order is None:

        hue_order = data[hue].unique()

    data[hue] = pd.Categorical(data[hue], hue_order, ordered=True)
    # data.sort_values([x, hue], inplace=True)

    # plot
    data = (
        data.groupby(x)
        .apply(lambda x: set_bottom(x, direction, center))
        .reset_index(drop=True)
    )

    data.sort_values(x, inplace=True)

    if isinstance(cmap, str):
        n_colors = data[hue].nunique()

        color_pallete = sns.color_palette(cmap, n_colors=n_colors)
        color_pallete = dict(zip(hue_order, color_pallete))
    elif isinstance(cmap, list):
        color_pallete = cmap
        color_pallete = dict(zip(hue_order, color_pallete))
    elif isinstance(cmap, dict):
        color_pallete = cmap
    else:
        raise ValueError("camp should be a list or a string")
    bar_legend_list = []
    for i, (name, group) in enumerate(data.groupby(hue)):
        c_color = color_pallete[name]
        bar = ax.bar(
            group[x],
            group[direction],
            bottom=group["bottom"],
            label=name,
            width=1,
            align="edge",
            edgecolor="black",
            linewidth=min(1 / (len(order) / 150), 1),
            color=c_color,
            **kwargs,
        )
        bar_legend = mpatches.Patch(color=c_color, label=name)
        bar_legend_list.append(bar_legend)

    ax.set_xticklabels([])
    ax.legend(
        handles=bar_legend_list,
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
        title=hue,
        ncol=np.sqrt(len(bar_legend_list)).astype(int),
    )
    # plt.legend()

    # ax.tick_params(axis="x", which="both", bottom=False, top=False)
    # plt.setp(ax.get_xticklabels(), rotation=90, ha="right", va="top")
    # annotate
    annotate_kwargs_default = {
        "topk": 10,
        "space_range": 0.2,
        "angle_range": 30,
        "left_texts_fraction": (0.05, 0.3),
        "right_texts_fraction": (0.75, 0.7),
        "fontsize": 6,
    }
    if annotate_kwargs is not None:
        annotate_kwargs_default.update(annotate_kwargs)

    topk = annotate_kwargs_default["topk"]
    space = annotate_kwargs_default["space_range"] / topk
    angle_space = annotate_kwargs_default["angle_range"] / topk

    left_texts_fraction = annotate_kwargs_default["left_texts_fraction"]
    fontsize = annotate_kwargs_default["fontsize"]

    # set the  left topk
    left_topk = order[:topk]
    left_texts = []
    for i, each in enumerate(left_topk):
        sort_each_df = data[data[x] == each].sort_values("bottom")
        min_row = sort_each_df.iloc[0]
        if min_row["direction"] < 0:
            each_y = min_row["bottom"] + min_row[direction]
        else:
            each_y = min_row["bottom"]

        # left_texts.append(ax.text(each, each_y, each, ha="center", va="top"))

        text = ax.annotate(
            each,
            (each, each_y - 0.1),
            xytext=(i * space + left_texts_fraction[0], left_texts_fraction[1]),
            xycoords="data",
            textcoords="axes fraction",
            ha="left",
            va="top",
            arrowprops=dict(
                arrowstyle="-",
                color="black",
                lw=1,
                # connectionstyle=f"angle, angleA={130 + angle_space *i},angleB=-90",
            ),
            rotation=-45,
            rotation_mode="anchor",
            fontsize=fontsize,
        )
        left_texts.append(text)

    right_topk = order[-topk:]
    right_texts = []
    right_texts_fraction = annotate_kwargs_default["right_texts_fraction"]

    for i, each in enumerate(right_topk):
        sort_each_df = data[data[x] == each].sort_values("bottom", ascending=False)
        max_row = sort_each_df.iloc[0]

        each_y = min_row["bottom"]

        # right_texts.append(ax.text(each, each_y, each, ha="center", va="top"))
        text = ax.annotate(
            each,
            (each, each_y + 0.1),
            xytext=(right_texts_fraction[0] + i * space, right_texts_fraction[1]),
            xycoords="data",
            textcoords="axes fraction",
            ha="right",
            va="bottom",
            arrowprops=dict(
                arrowstyle="-",
                color="black",
                lw=1,
                # connectionstyle=f"angle, angleA={-30 - i * angle_space},angleB=90",
            ),
            rotation=-45,
            fontsize=fontsize,
            rotation_mode="anchor",
        )
        right_texts.append(text)

    ax.tick_params(axis="x", which="both", bottom=False, top=False)

    return ax
