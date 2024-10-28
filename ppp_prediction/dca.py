from dcurves import dca
import numpy as np


def get_dca_data(
    data,
    outcome,
    model_cols,
    threshold=None,
    steps=100,
):
    data = data[[outcome] + model_cols].copy().dropna()
    if threshold is None:
        threshold = np.linspace(0, 1, 500)

        dca_famhistory_df = dca(
            data=data,
            outcome=outcome,
            modelnames=model_cols,
            thresholds=threshold,
        )

        dca_famhistory_df["st_net_benefit"] = (
            dca_famhistory_df["net_benefit"] / dca_famhistory_df["prevalence"]
        )

        threshold = dca_famhistory_df.query("st_net_benefit > 0 ")["threshold"].max()
        print(f"threshold is {threshold} and sampling {steps} points")
        return get_dca_data(
            data=data,
            outcome=outcome,
            model_cols=model_cols,
            threshold=threshold,
            steps=steps,
        )

    else:
        dca_famhistory_df = dca(
            data=data,
            outcome=outcome,
            modelnames=model_cols,
            thresholds=np.linspace(0, threshold, steps),
        )

        dca_famhistory_df["st_net_benefit"] = (
            dca_famhistory_df["net_benefit"] / dca_famhistory_df["prevalence"]
        )

        return dca_famhistory_df


# dca_df["model"].unique()
"""
Code From: https://github.com/MSKCC-Epi-Bio/dcurves
Modified by Tingfeng Xu
TODO: sns.lineplot to update the ax.plot 
This module houses plotting functions used in the user-facing plot_graphs() 
function to plot net-benefit scores and net interventions avoided.
"""

from typing import Optional, Iterable
import random
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from numpy import ndarray
import seaborn as sns


def _plot_net_benefit(
    plot_df: pd.DataFrame,
    y_limits: Iterable = (-0.05, 1),
    color_names: Iterable = None,
    show_grid: bool = True,
    show_legend: bool = True,
    smoothed_data: Optional[dict] = None,  # Corrected parameter
    ax=None,
    line_kwargs=None,
) -> None:
    """
    Plot net benefit values against threshold probability values. Can use pre-computed smoothed data if provided.

    Parameters
    ----------
    plot_df : pd.DataFrame
        Data containing threshold probability values and model columns of net benefit scores to be plotted.
    y_limits : Iterable[float], optional
        Tuple or list with two floats specifying the y-axis limits.
    color_names : Iterable[str], optional
        List of colors for each model line. Must match the number of unique models if provided.
    show_grid : bool, optional
        If True, display grid lines on the plot. Default is True.
    show_legend : bool, optional
        If True, display the legend on the plot. Default is True.
    smoothed_data : dict, optional
        Pre-computed smoothed data for each model. Keys are model names, and values are arrays with smoothed points.

    Raises
    ------
    ValueError
        If the input dataframe does not contain the required columns or if y_limits or color_names are incorrectly formatted.

    Returns
    -------
    None
    """

    # Validate input dataframe
    required_columns = ["threshold", "model", "net_benefit"]
    if not all(column in plot_df.columns for column in required_columns):
        raise ValueError(
            f"plot_df must contain the following columns: {', '.join(required_columns)}"
        )

    # Validate y_limits
    if len(y_limits) != 2 or y_limits[0] >= y_limits[1]:
        raise ValueError(
            "y_limits must contain two floats where the first is less than the second"
        )

    # Validate color_names
    modelnames = plot_df["model"].unique()
    if isinstance(color_names, list) and len(color_names) != len(modelnames):

        raise ValueError(
            "The length of color_names must match the number of unique models"
        )

    # Plotting
    for idx, modelname in enumerate(plot_df["model"].unique()):
        if isinstance(color_names, list):
            color = color_names[idx]  # Directly use color from color_names by index
        elif isinstance(color_names, dict):
            color = color_names[modelname]
        else:
            raise ValueError("color_names must be a list or a dictionary")
        model_df = plot_df[plot_df["model"] == modelname]
        if smoothed_data and modelname in smoothed_data:
            smoothed = smoothed_data[modelname]
            if not isinstance(smoothed, ndarray):
                raise ValueError(
                    f"Smoothed data for '{modelname}' must be a NumPy array."
                )
            ax.plot(
                smoothed[:, 0],
                smoothed[:, 1],
                color=color,
                label=modelname,
                **line_kwargs,
            )
        else:
            ax.plot(
                model_df["threshold"],
                model_df["net_benefit"],
                color=color,
                label=modelname,
                **line_kwargs,
            )
    ax.set_ylim(y_limits)

    if show_legend:
        ax.legend()
    if show_grid:
        ax.grid(color="black", which="both", axis="both", linewidth="0.3")
        # plt.grid(color="black", which="both", axis="both", linewidth="0.3")
    else:
        ax.grid(False)
    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")


def _plot_net_intervention_avoided(
    plot_df: pd.DataFrame,
    y_limits: Iterable = (-0.05, 1),
    color_names: Iterable = None,
    show_grid: bool = True,
    show_legend: bool = True,
    smoothed_data: Optional[dict] = None,  # Updated to accept smoothed data
    ax=None,
    line_kwargs=None,
) -> None:
    """
    Plot net interventions avoided values against threshold probability values. Can use pre-computed smoothed data if provided.

    Parameters
    ----------
    plot_df : pd.DataFrame
        Data containing threshold probability values and model columns of net interventions avoided scores to be plotted.
    y_limits : Iterable[float]
        Tuple or list with two floats specifying the y-axis limits.
    color_names : Iterable[str]
        List of colors for each model line. Must match the number of unique models if provided.
    show_grid : bool
        If True, display grid lines on the plot. Default is True.
    show_legend : bool
        If True, display the legend on the plot. Default is True.
    smoothed_data : dict, optional
        Pre-computed smoothed data for each model. Keys are model names, and values are arrays with smoothed points.

    Raises
    ------
    ValueError
        If the input dataframe does not contain the required columns or if y_limits or color_names are incorrectly formatted.

    Returns
    -------
    None
    """

    # Validate input dataframe
    required_columns = ["threshold", "model", "net_intervention_avoided"]
    if not all(column in plot_df.columns for column in required_columns):
        raise ValueError(
            f"plot_df must contain the following columns: {', '.join(required_columns)}"
        )

    # Validate y_limits
    if len(y_limits) != 2 or y_limits[0] >= y_limits[1]:
        raise ValueError(
            "y_limits must contain two floats where the first is less than the second"
        )

    # Validate color_names
    modelnames = plot_df["model"].unique()
    if color_names and len(color_names) != len(modelnames):
        raise ValueError(
            "The length of color_names must match the number of unique models"
        )

    # Plotting
    for idx, modelname in enumerate(plot_df["model"].unique()):
        color = color_names[idx]  # Directly use color from color_names by index
        model_df = plot_df[plot_df["model"] == modelname]
        if model_df.empty:  # Skip plotting for empty DataFrames
            continue
        if smoothed_data and modelname in smoothed_data:
            smoothed = smoothed_data[modelname]
            if smoothed_data and modelname in smoothed_data:
                if not isinstance(smoothed, ndarray):
                    raise ValueError(
                        f"Smoothed data for '{modelname}' must be a NumPy array."
                    )
            ax.plot(
                smoothed[:, 0],
                smoothed[:, 1],
                color=color,
                label=modelname,
                **line_kwargs,
            )
        else:
            ax.plot(
                model_df["threshold"],
                model_df["net_intervention_avoided"],
                color=color,
                label=modelname,
                **line_kwargs,
            )
    ax.set_ylim(y_limits)

    if show_legend:
        ax.legend()
    if show_grid:
        ax.grid(color="black", which="both", axis="both", linewidth="0.3")
    else:
        ax.grid(False)
    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Reduction of Interventions")


def plot_graphs(
    plot_df: pd.DataFrame,
    graph_type: str = "net_benefit",
    y_limits: Iterable = (-0.05, 1),
    color_names: Optional[Iterable] = None,
    show_grid: bool = True,
    show_legend: bool = True,
    smooth_frac: float = 0.0,  # Default to 0, indicating no smoothing unless specified
    file_name: Optional[str] = None,
    dpi: int = 100,
    ax=None,
    line_kwargs=None,
) -> None:
    """
    Plot specified graph type for the given data, either net benefit or net interventions avoided,
    against threshold probabilities. Applies LOWESS smoothing if `smooth_frac` is greater than 0,
    excluding 'all' and 'none' models from smoothing. The smoothing will be more sensitive to local variations,
    keeping the smoothed lines closer to the original data points if `smooth_frac` is specified.

    Parameters
    ----------
    plot_df : pd.DataFrame
        DataFrame containing 'threshold', 'model', and either 'net_benefit' or 'net_intervention_avoided' columns.
    graph_type : str, optional
        Specifies the type of plot to generate. Valid options are 'net_benefit' or 'net_intervention_avoided'.
    y_limits : Iterable[float], optional
        Two-element iterable specifying the lower and upper bounds of the y-axis.
    color_names : Iterable[str], optional
        List of colors to use for each line in the plot. Must match the number of models in `plot_df`.
    show_grid : bool, optional
        If True, display grid lines on the plot. Default is True.
    show_legend : bool, optional
        If True, display the legend on the plot. Default is True.
    smooth_frac : float, optional
        Fraction of data points used when estimating each y-value in the smoothed line,
        making the smoothing more sensitive to local variations. Set to 0 for no smoothing. Default is 0.
    file_name : str, optional
        Path and file name where the figure will be saved. If None, the figure is not saved.
    dpi : int, optional
        Resolution of the saved figure in dots per inch.

    Raises
    ------
    ValueError
        If `graph_type` is not recognized.
        If `y_limits` does not contain exactly two elements or if the lower limit is not less than the upper limit.
        If `color_names` is provided but does not match the number of models in `plot_df`.
        If `smooth_frac` is not within the 0 to 1 range.
        If the input DataFrame is empty.

    Returns
    -------
    None
    """
    if ax is None:
        fig, ax = plt.subplots()
    if plot_df.empty:
        raise ValueError("The input DataFrame is empty.")

    if graph_type not in ["net_benefit", "net_intervention_avoided"]:
        raise ValueError(
            "graph_type must be 'net_benefit' or 'net_intervention_avoided'"
        )

    if len(y_limits) != 2 or y_limits[0] >= y_limits[1]:
        raise ValueError(
            "y_limits must contain two floats where the first is less than the second"
        )

    if not 0 <= smooth_frac <= 1:
        raise ValueError("smooth_frac must be between 0 and 1")

    modelnames = plot_df["model"].unique()
    # set color
    if color_names is None:
        color_names = "Set1"

    if isinstance(color_names, str):
        color_names = sns.palettes.color_palette(color_names, len(modelnames))

    if len(color_names) < len(modelnames):
        raise ValueError(
            "color_names must match the number of unique models in plot_df"
        )

    smoothed_data = {}
    if smooth_frac > 0:  # Apply smoothing only if smooth_frac is greater than 0
        lowess = sm.nonparametric.lowess
        for modelname in plot_df["model"].unique():
            # Skip 'all' and 'none' models from smoothing
            if modelname.lower() in ["all", "none"]:
                continue

            model_df = plot_df[plot_df["model"] == modelname]
            y_col = (
                "net_benefit"
                if graph_type == "net_benefit"
                else "net_intervention_avoided"
            )
            smoothed_data[modelname] = lowess(
                model_df[y_col], model_df["threshold"], frac=smooth_frac
            )

    plot_function = (
        _plot_net_benefit
        if graph_type == "net_benefit"
        else _plot_net_intervention_avoided
    )
    plot_function(
        plot_df=plot_df,
        y_limits=y_limits,
        color_names=color_names,
        show_grid=show_grid,
        show_legend=show_legend,
        smoothed_data=(
            smoothed_data if smooth_frac > 0 else None
        ),  # Pass smoothed_data only if smoothing was applied
        ax=ax,
        line_kwargs=line_kwargs,
    )

    if file_name:
        try:
            plt.savefig(file_name, dpi=dpi)
        except Exception as e:
            print(f"Error saving figure: {e}")


def plot_dca(
    data,
    threshold="threshold",
    net_benefit="st_net_benefit",
    method_col="Method",
    fill_between=None,
    line_kwargs=None,
    ax=None,
    smooth_frac=0.05,
    palette=None,
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots()

    data = data.copy().rename(columns={method_col: "model"})
    data["net_benefit"] = data[net_benefit]

    method_col = "model"
    line_kwargs_default = {"lw": 0.8}
    if line_kwargs is not None:
        line_kwargs_default.update(line_kwargs)
    if palette is None:
        palette = sns.color_palette("Set2", len(data[method_col].unique()))
        palette = {k: v for k, v in zip(data[method_col].unique(), palette)}
        palette["none"] = "grey"
        palette["all"] = "darkgrey"
    elif isinstance(palette, str):
        palette = sns.color_palette(palette, len(data[method_col].unique()))
        palette = {k: v for k, v in zip(data[method_col].unique(), palette)}
        palette["none"] = "grey"
        palette["all"] = "darkgrey"
    elif isinstance(palette, dict):
        palette = palette
        if "none" not in palette:
            palette["none"] = "grey"
        if "all" not in palette:
            palette["all"] = "darkgrey"
    elif isinstance(palette, list):
        palette = {k: v for k, v in zip(data[method_col].unique(), palette)}
        palette["none"] = "grey"
        palette["all"] = "darkgrey"
    else:
        raise ValueError("palette must be a list, dict or str")

    plot_graphs(
        plot_df=data,
        graph_type="net_benefit",
        y_limits=[0, data[net_benefit].max()],
        ax=ax,
        line_kwargs=line_kwargs_default,
        show_grid=False,
        color_names=palette,
    )

    if fill_between is not None:
        for idx, (ref, target) in enumerate(fill_between):
            thresholds = [i.get_data()[0] for i in ax.lines if i.get_label() == ref][0]

            ref_line = [i.get_data()[1] for i in ax.lines if i.get_label() == ref][0]
            target_line = [
                i.get_data()[1] for i in ax.lines if i.get_label() == target
            ][0]

            ax.fill_between(
                thresholds,
                ref_line,
                target_line,
                where=(target_line > ref_line),
                alpha=0.3,
                label=f"{target} vs {ref}",
                color=palette[target],
            )

    ax.legend(prop={"size": 10})

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Threshold probability (%)")
    ax.set_ylabel("Standardized Net benefit (%)")
    ax.set_yticklabels(
        [f"{i*100:.0f}" for i in ax.get_yticks()],
        fontdict={"color": "#4e4d4e", "family": "Calibri"},
    )
    ax.set_xticklabels(
        [f"{i*100:.1f}" for i in ax.get_xticks()],
        fontdict={"color": "#4e4d4e", "family": "Calibri"},
    )

    legend = ax.legend(
        # handles,
        # labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=4,  # lager size
        prop={"size": 12},
        frameon=False,
        handlelength=2,
    )
    for line in legend.get_lines():
        line.set_linewidth(3)

    return ax
