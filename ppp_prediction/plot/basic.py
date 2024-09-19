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


def plot_scatter_diag_style1(data, x, y, color=None, text=None, cmap=None):
    data = data.copy()

    if cmap is None:
        cmap = sns.color_palette("tab20", n_colors=len(data[color].unique()))

    p = (
        ggplot(data=data, mapping=aes(x=x, y=y, color=color, text=text))
        + geom_point(size=4)
        + geom_abline(intercept=0, slope=1, linetype="dashed")
        + geom_text(
            mapping=aes(label=text),
            nudge_x=0.01,
            nudge_y=0.01,
        )
        + theme_classic(base_family="Calibri", base_size=12)  # 使用Tufte主题
        + theme(axis_line=element_line())
        + theme(
            figure_size=(6, 6),
            legend_position="top",
            axis_text_x=element_text(angle=90),
            strip_background=element_blank(),
            axis_text=element_text(size=12),  # 调整轴文字大小
            axis_title=element_text(size=14),  # 调整轴标题大小和样式
            legend_title=element_text(size=14),  # 调整图例标题大小和样式
            legend_text=element_text(),  # 调整图例文字大小
            strip_text=element_text(size=14),  # 调整分面标签的大小和样式
            plot_title=element_text(size=16, hjust=0.5),  # 添加图表标题并居中
            # plot_margin = margin(10, 10, 10, 10)  # 设置图表边距
        )
        + scale_color_manual(
            # values=list(color_dict.values()),
            values=cmap
        )
    )
    return p