from sklearn.metrics import roc_curve, auc
import pandas as pd
from sklearn.metrics import auc
import seaborn as sns
import matplotlib.pyplot as plt

def get_auc_df(
    data,
    label,
    method_cols,
):
    """
    get the auc df
    """
    auc_list = []
    for method_col in method_cols:
        fpr, tpr, _ = roc_curve(data[label], data[method_col])
        roc_current_df = pd.DataFrame(
            [
                {
                    "fpr": fpr_,
                    "tpr": tpr_,
                }
                for fpr_, tpr_ in zip(fpr, tpr)
            ]
        )
        roc_current_df['Method'] = method_col
        auc_list.append(roc_current_df)

    return pd.concat(auc_list)


def plot_auc_v1(
    data,
    FPR="fpr",
    TPR="tpr",
    Method_col="Method",
    palette=None,
    ax=None,
    lw=3,
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    sns.lineplot(
        data=data,
        x=FPR,
        y=TPR,
        lw=lw,
        hue=Method_col,
        palette=palette,
        estimator=None,
        ax=ax,
    )
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=2)

    ax.set_xlabel("1 - Specificity", fontsize=18)
    ax.set_ylabel("Sensitivity", fontsize=18)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(labelsize=16)
    # ax.xais.set_params(lw=2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)

    roc_auc_df = (
        data.groupby([Method_col])
        .apply(lambda x: pd.Series({"AUC": round(auc(x[FPR], x[TPR]), 3)}))
        .reset_index()
    )

    roc_auc_df["text"] = roc_auc_df[Method_col] + ": " + roc_auc_df["AUC"].astype(str)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles,
        labels=[
            f"{l}: {roc_auc_df.query('Method == @l')['AUC'].values[0]:.3f}"
            for l in labels
        ],
        fontsize=16,
        title_fontsize=16,
        loc="lower right",
    )
    return ax