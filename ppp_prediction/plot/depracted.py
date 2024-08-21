import pandas as pd
import matplotlib.pyplot as plt


def plot_table(df, bbox, loc, ax, table_style, table_style_config=None, title=None):
    """
    Plot a table using the given DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data to be plotted.
    - bbox (list): The bounding box coordinates of the table as [left, bottom, width, height].
    - loc (str): The location of the table within the plot. Possible values are 'center', 'upper right', 'upper left', 'lower right', 'lower left', 'upper center', 'lower center', 'center right', 'center left'.
    - ax (matplotlib.axes.Axes): The axes object on which to plot the table.
    - table_style (str): The style of the table. Possible values are 'nejm', 'nature', 'cell'.
    - table_style_config (dict, optional): Additional configuration for the table style. Defaults to None.

    Example:
    # Usage example
    fig, ax = plt.subplots(figsize=(10, 6))
    data = {
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [25, 30, 35],
        "Job": ["Engineer", "Doctor", "Artist"],
    }
    df = pd.DataFrame(data)
    plot_table(df, bbox=[0.9, 0, 0.3, 0.2], loc="center", ax=ax, table_style="nature")
    plt.show()
    """
    # Predefined style configuration dictionary
    style_configs = {
        "nejm": {
            "fontsize": 12,
            "fontweight": "bold",
            "col_bg_color": "#f1f1f1",
            "cell_bg_color": "white",
            "linewidth": 0.5,
            "title_size": 16,
        },
        "nature": {
            "fontsize": 10,
            "fontweight": "normal",
            "col_bg_color": "#d9d9d9",
            "cell_bg_color": "white",
            "linewidth": 0.8,
            "title_size": 14,
        },
        "cell": {
            "fontsize": 11,
            "fontweight": "normal",
            "col_bg_color": "#e6e6e6",
            "cell_bg_color": "white",
            "linewidth": 0.6,
            "title_size": 15,
        },
    }

    # Get the configuration for the specified table style
    config = style_configs.get(table_style)
    if table_style_config is not None:
        config.update(table_style_config)
    # Create the table
    the_table = ax.table(cellText=df.values, colLabels=df.columns, loc=loc, bbox=bbox)

    # Set the table style
    the_table.auto_set_font_size(True)
    the_table.set_fontsize(config["fontsize"])
    the_table.auto_set_column_width(
        col=list(range(len(df.columns)))
    )  # Automatically set column width
    for key, cell in the_table.get_celld().items():
        cell.set_linewidth(config["linewidth"])
        if key[0] == 0:
            cell.set_fontsize(config["fontsize"])  # Column header font size
            cell.set_facecolor(config["col_bg_color"])  # Column header background color
            cell.set_text_props(weight=config["fontweight"])  # Column header font style
        else:
            cell.set_facecolor(config["cell_bg_color"])

    # Turn off the axis
    ax.axis("off")

    # Set the title
    if title:
        ax.set_title(title, fontweight="bold", size=config["title_size"], pad=20)
    return ax
