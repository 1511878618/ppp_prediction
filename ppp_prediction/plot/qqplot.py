import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from adjustText import adjust_text


def simple_annotate(
    ax,
    points,
    annotations,
    repel_force=0.02,
    fontsize=10,
    arrow_color="grey",
    text_color="black",
):
    """
    在图表上标注点，带有自动调整位置的功能。

    参数：
        - ax: matplotlib Axes 对象
        - points: list of tuples, 每个点的坐标列表 [(x1, y1), (x2, y2), ...]
        - annotations: list of str, 每个点的注释文字
        - repel_force: float, 自动调整文本位置的排斥力，值越大文本越分散
        - fontsize: int, 注释文字大小
        - arrow_color: str, 箭头颜色
        - text_color: str, 注释文字颜色
    """
    texts = []

    for (x, y), annotation in zip(points, annotations):
        text = ax.text(
            x,
            y + 0.05,
            annotation,
            fontsize=fontsize,
            color=text_color,
            ha="center",
            va="bottom",
            zorder=10,
        )
        texts.append(text)

        ax.annotate(
            "",
            xy=(x, y),
            xytext=(x, y + 0.05),
            arrowprops=dict(arrowstyle="->", color=arrow_color, linewidth=0.8),
        )

    adjust_text(
        texts,
        only_move={"points": "y", "text": "xy"},
        force_text=repel_force,
        force_points=repel_force,
        expand_text=(1.2, 1.2),
        expand_points=(1.2, 1.2),
        ax=ax,
    )


def plot_gwas_qq(df, pvalue_col, anno_col=None, topk=10, title="QQ Plot", ax=None):
    """
    绘制 GWAS QQ 图，并对指定列中的 Top K 个点进行标注。

    参数：
        - df: pandas.DataFrame, 包含 p 值和标注信息的数据框
        - pvalue_col: str, 指定 p 值所在的列名
        - anno_col: str, 可选，指定标注信息所在的列名。如果为 None，则用索引作为标注信息
        - topk: int, 标注的前 K 个点
        - title: str, 图标题
        - ax: matplotlib Axes 对象，默认为 None，将自动创建
    """
    # 提取 p 值
    pvalues = df[pvalue_col].dropna()
    pvalues = pvalues[pvalues > 0]  # 避免 log(0)

    # 计算预期和观察的 -log10(p) 值
    n = len(pvalues)
    expected = -np.log10((np.arange(1, n + 1) - 0.5) / n)
    observed = -np.log10(np.sort(pvalues))

    # 计算 Lambda GC
    chi2_obs = chi2.isf(pvalues, df=1)
    lambda_gc = np.median(chi2_obs) / chi2.ppf(0.5, df=1)

    # 创建图形
    if ax is None:
        figsize = (6, 6)
        fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(expected, observed, color="steelblue", alpha=0.8, s=10, label="Observed")
    ax.plot(
        [expected.min(), expected.max()],
        [expected.min(), expected.max()],
        color="gray",
        linestyle="--",
        label="y = x",
    )

    # 添加 Lambda GC
    ax.text(
        0.05,
        0.95,
        f"$\\lambda = {lambda_gc:.4f}$",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="left",
    )

    # 设置标题和轴标签
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Expected $-\\log_{10}(P)$", fontsize=12)
    ax.set_ylabel("Observed $-\\log_{10}(P)$", fontsize=12)
    ax.grid(alpha=0.3)

    # 标注 Top K 个点
    topk_idx = np.argsort(df[pvalue_col])[:topk]
    topk_pvalues = df.iloc[topk_idx][pvalue_col]
    topk_observed = -np.log10(topk_pvalues)
    topk_expected = expected[:topk]

    # 如果没有 anno_col，则使用索引作为标注
    if anno_col:
        topk_annotations = df.iloc[topk_idx][anno_col].values
    else:
        topk_annotations = [str(i) for i in topk_idx]

    # 将 Top K 点的坐标与标注传入 simple_annotate
    topk_points = list(zip(topk_expected, topk_observed))
    simple_annotate(ax, topk_points, topk_annotations)

    plt.tight_layout()
    return ax


# 测试数据
if __name__ == "__main__":
    import pandas as pd

    # 创建测试数据
    np.random.seed(42)
    data = {
        "pvalue": np.random.uniform(0, 1, 1000),
        "snp": [f"SNP{i}" for i in range(1000)],
    }
    df = pd.DataFrame(data)

    # 调用函数绘制 QQ 图
    plot_gwas_qq(df, pvalue_col="pvalue", anno_col="snp", topk=10)
    plt.show()