import seaborn.objects as so
import matplotlib.pyplot as plt
import seaborn as sns
from pypalettes import load_cmap


def plot_stack_hist_v1(
    data,
    x,
    hue,
    ax=None,
    palette=None,
    hue_order=None,
    # title=None,
):
    """
    Plot a stacked histogram with percentage values sum to 1 for each x value.

    Args:
        data (pandas.DataFrame): The input data.
        x (str): The column name for the x-axis.
        hue (str): The column name for the hue (color) of the bars.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. If not provided, a new figure and axes will be created.
        palette (list, optional): The color palette to use for the bars. If not provided, a default palette will be used.

    Returns:
        matplotlib.axes.Axes: The plotted axes.

    Example:
        tips = sns.load_dataset("tips")

        plot_stack_hist_v1(
            data=tips,
            x="size",
            hue="day",
        )

    """
    data = data.copy()
    if hue:
        data = data.dropna(subset=[hue])
        if not hue_order:
            hue_order = data[hue].unique()
        print(data[hue].isna().sum())
        data[hue] = (
            data[hue].astype("category").cat.set_categories(hue_order, ordered=True)
        )
    grouped = data.groupby([x, hue]).size().reset_index(name="counts")
    group_sums = grouped.groupby(x)["counts"].transform("sum")
    grouped["percentage"] = grouped["counts"] / group_sums * 100
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 6))
    if palette is None:
        palette = load_cmap("Lupi").colors
    p = (
        so.Plot(
            grouped,
            x=x,
            y="percentage",
            color=hue,
        )
        .add(so.Bar(alpha=0.6, edgecolor="black", edgewidth=0.3), so.Stack())
        .scale(color=palette)
        .on(ax)
        .plot()
    )
    fig = ax.get_figure()
    # legend = fig.legends[0]
    legend = fig.legends.pop(0)
    ax.legend(
        legend.legend_handles,
        [t.get_text() for t in legend.texts],
        # loc="upper right",
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=2,
        title=hue,
        frameon=True,
    )

    frame = ax.get_legend().get_frame()
    frame.set_edgecolor("grey")
    frame.set_linewidth(0.5)
    ax.set_xlabel(x)
    ax.set_ylabel("Percentage")
    return ax

def plot_stack_hist_v2(
    data,
    x,
    hue,
    y=None,
    ax=None,
    palette=None,
    hue_order=None,
    # title=None,
):
    """
    Plot a stacked histogram with percentage values sum to 1 for each x value.

    Args:
        data (pandas.DataFrame): The input data.
        x (str): The column name for the x-axis.
        hue (str): The column name for the hue (color) of the bars.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. If not provided, a new figure and axes will be created.
        palette (list, optional): The color palette to use for the bars. If not provided, a default palette will be used.

    Returns:
        matplotlib.axes.Axes: The plotted axes.

    Example:
        tips = sns.load_dataset("tips")

        plot_stack_hist_v1(
            data=tips,
            x="size",
            hue="day",
        )

    """
    data = data.copy()
    if hue:
        data = data.dropna(subset=[hue])
        if not hue_order:
            hue_order = data[hue].unique()
        print(data[hue].isna().sum())
        data[hue] = (
            data[hue].astype("category").cat.set_categories(hue_order, ordered=True)
        )
    if y is None:

        grouped = data.groupby([x, hue]).size().reset_index(name="counts")
        group_sums = grouped.groupby(x)["counts"].transform("sum")
        grouped["percentage"] = grouped["counts"] / group_sums * 100
        y = "percentage"
    else:
        grouped = data
        y = y
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 6))
    if palette is None:
        palette = load_cmap("Lupi").colors
    p = (
        so.Plot(
            grouped,
            x=x,
            y=y,
            color=hue,
        )
        .add(so.Bar(alpha=0.6, edgecolor="black", edgewidth=0.3), so.Stack())
        .scale(color=palette)
        .on(ax)
        .plot()
    )
    fig = ax.get_figure()
    # legend = fig.legends[0]
    legend = fig.legends.pop(0)
    ax.legend(
        legend.legend_handles,
        [t.get_text() for t in legend.texts],
        # loc="upper right",
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=2,
        title=hue,
        frameon=True,
    )

    frame = ax.get_legend().get_frame()
    frame.set_edgecolor("grey")
    frame.set_linewidth(0.5)
    ax.set_xlabel(x)
    ax.set_ylabel("Percentage")
    return ax