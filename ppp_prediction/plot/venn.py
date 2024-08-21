import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted, venn3, venn3_circles


def plot_venn2(
    Group1,
    Group2,
    ax=None,
    labels=("Group1", "Group2"),
    colors=("#0073C2FF", "#EFC000FF"),
    title=None,
    alpha=0.8,
):
    if ax is None:
        fig, ax = plt.subplots()

    if title is None:
        title = f"Venn Diagram of {labels[0]} and {labels[1]}"
    vee2 = venn2(
        [set(Group1), set(Group2)],
        # set_labels=("Group1", "Group2"),
        set_labels=labels,
        set_colors=colors,
        # set_colors=("#0073C2FF", "#EFC000FF"),
        alpha=alpha,
        ax=ax,
    )
    venn2_circles(
        [set(Group1), set(Group2)], linestyle="--", linewidth=2, color="black", ax=ax
    )

    # 定制化设置:设置字体属性
    for text in vee2.set_labels:
        text.set_fontsize(15)
    for text in vee2.subset_labels:
        try:
            text.set_fontsize(16)
            text.set_fontweight("bold")
        except:
            pass
    # ax.text(
    #     0.8,
    #     -0.1,
    #     "\nVisualization by DataCharm",
    #     transform=ax.transAxes,
    #     ha="center",
    #     va="center",
    #     fontsize=8,
    #     color="black",
    # )
    ax.set_title(
        title,
        size=15,
        fontweight="bold",
        # backgroundcolor="#868686FF",
        color="black",
        style="italic",
    )
    return ax


def plot_venn3(
    Group1,
    Group2,
    Group3,
    ax=None,
    labels=("Group1", "Group2", "Group3"),
    colors=("#0073C2FF", "#EFC000FF", "#CD534CFF"),
    title=None,
):
    if ax is None:
        fig, ax = plt.subplots()
    if title is None:
        title = f"Venn Diagram of {labels[0]}, {labels[1]} and {labels[2]}"
    vd3 = venn3(
        [set(Group1), set(Group2), set(Group3)],
        set_labels=labels,
        set_colors=colors,
        alpha=0.8,
        ax=ax,
    )
    venn3_circles(
        [set(Group1), set(Group2), set(Group3)],
        linestyle="--",
        linewidth=2,
        color="black",
        ax=ax,
    )
    for text in vd3.set_labels:
        text.set_fontsize(15)
        text.set_fontweight("bold")
    for text in vd3.subset_labels:
        try:
            text.set_fontsize(15)
            text.set_fontweight("bold")
        except:
            pass
    # ax.text(
    #     0.8,
    #     -0.1,
    #     "\nVisualization by DataCharm",
    #     transform=ax.transAxes,
    #     ha="center",
    #     va="center",
    #     fontsize=9,
    #     color="black",
    # )
    ax.set_title(
        title,
        fontweight="bold",
        fontsize=15,
        pad=30,
        # backgroundcolor="#868686FF",
        color="black",
        style="italic",
    )
    return ax
