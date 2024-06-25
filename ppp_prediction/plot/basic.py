import seaborn as sns
import matplotlib.pyplot as plt


def plot_missing(df, columns, ax=None, bar_kwargs=None, max_show_k=20):
    missing = df[columns].isna().sum() / df.shape[0]
    missing = missing.sort_values(ascending=False)

    if missing.shape[0] >= 20:
        print("Too many columns to plot")
        if max_show_k is not None:
            missing = missing.head(max_show_k)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(1 * missing.shape[0], 5))

    bar_kwargs_default = dict(palette="Set2", edgecolor="black")
    if bar_kwargs is not None:
        bar_kwargs_default.update(bar_kwargs)
    # sns.barplot(kind="bar", ax=ax, **bar_kwargs_default)
    sns.barplot(x=missing.index, y=missing.values, ax=ax, **bar_kwargs_default)
    ax.set_ylabel("Missing rate", fontsize=16)
    ax.set_xlabel("Columns", fontsize=16)
    ax.set_title("Missing rate of columns", fontsize=16)
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=12, direction="out")
