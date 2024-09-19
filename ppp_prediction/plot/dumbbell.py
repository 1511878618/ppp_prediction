from plotnine import *
import pandas as pd


# 哑铃图
def dumbbell_plot(
    data,
    x,
    y,
    group_col,
    ref_group,
    new_group,
    color_dict=None,
    figsize=(6, 6),
    **kwargs,
):
    """
    Generate a dumbbell plot using the given data.
    Parameters:
    - data: DataFrame
        The input data containing the necessary columns.
    - x: str
        The column name for the x-axis.
    - y: str
        The column name for the y-axis.
    - group_col: str
        The column name for grouping the data.
    - ref_group: str
        The reference group for comparison of group_col
    - new_group: str
        The new group for comparison of group_col
    - color_dict: dict
        A dictionary mapping group names to color values.
    - figsize: tuple, optional
        The size of the figure (default is (6, 6)).
    - **kwargs: optional
        Additional keyword arguments to be passed to the plot.
    Returns:
    - p: ggplot object
        The dumbbell plot.

    """
    if color_dict is None:
        color_dict = {ref_group: "#00a5ff", new_group: "#f26c64"}

    need_cols = [ref_group, new_group]
    plt_data1 = data.query(f"{group_col} in @need_cols")

    plt_data1[group_col] = pd.Categorical(
        plt_data1[group_col], categories=need_cols, ordered=True
    )

    plt_data = plt_data1.pivot_table(index=x, columns=group_col, values=y).reset_index(
        drop=False
    )

    plt_data["diff"] = plt_data[new_group] - plt_data[ref_group]
    plt_data = plt_data.sort_values("diff", ascending=False)
    plt_data[x] = pd.Categorical(
        plt_data[x], categories=plt_data[x].values[::-1], ordered=True
    )

    # x_order = plt_data[x].values.tolist()

    base_size = 2
    base_size = 2
    base_alpha = 0.6
    cutoff = 0.003

    gain_x = plt_data.query("diff > @cutoff")[x].tolist()
    decrease_x = plt_data.query("diff < @cutoff")[x].tolist()

    gain_df = plt_data1.query(f"{x} in @gain_x")
    decrease_df = plt_data1.query(f"{x} in @decrease_x")

    gain_df_segement = plt_data.query(f"{x} in @gain_x")
    decrease_df_segement = plt_data.query(f"{x} in @decrease_x")

    p = (
        ggplot(data=plt_data, mapping=aes(x=x, color=x))
        + geom_segment(
            data=gain_df_segement,
            mapping=aes(xend=x, y=ref_group, yend=new_group),
            color="#bebebe",
            size=base_size,
        )
        + geom_point(
            data=gain_df,
            mapping=aes(x=x, y=y, color=group_col),
            size=base_size * 2,
            alpha=base_alpha,
        )
        + geom_segment(
            data=decrease_df_segement,
            mapping=aes(xend=x, y=ref_group, yend=new_group),
            color="#bebebe",
            size=base_size,
            alpha=0.1,
        )
        + geom_point(
            data=decrease_df,
            mapping=aes(x=x, y=y, color=group_col),
            size=base_size * 2,
            alpha=0.1,
        )
        # + scale_color_manual(values=load_cmap("Classic_Cyclic").colors)
        + theme_classic(base_family="Calibri", base_size=12)  # 使用Tufte主题
        + theme(axis_line=element_line())
        + theme(
            figure_size=figsize,
            legend_position="top",
            axis_text_x=element_text(angle=90),
            strip_background=element_blank(),
            axis_text=element_text(size=12),  # 调整轴文字大小
            axis_title=element_text(size=14),  # 调整轴标题大小和样式
            legend_title=element_text(size=14),  # 调整图例标题大小和样式
            legend_text=element_text(size=14),  # 调整图例文字大小
            strip_text=element_text(size=14),  # 调整分面标签的大小和样式
            plot_title=element_text(size=16, hjust=0.5),  # 添加图表标题并居中
            # plot_margin = margin(10, 10, 10, 10)  # 设置图表边距
        )
        + scale_color_manual(values=color_dict)
        + guides(color=guide_legend(title=group_col))
        # + scale_color_discrete(breaks=[ref, new], labels=[ref, new])
        + labs(x=x, y=y)
        # + coord_flip()
    )

    return p
