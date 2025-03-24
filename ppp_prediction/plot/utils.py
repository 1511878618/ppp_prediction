import matplotlib.pyplot as plt
from pathlib import Path
from plotnine import ggplot

# def save_fig(fig=None, path=None, bbox_inches="tight", dpi=400, tiff=False, **kwargs):
#     if path is None:
#         path = "temp"
#     if fig is None:
#         fig = plt.gcf()
#     Path(path).parent.mkdir(parents=True, exist_ok=True)

#     fig.savefig(f"{path}.png", dpi=400, bbox_inches=bbox_inches, **kwargs)
#     fig.savefig(f"{path}.pdf", dpi=400, bbox_inches=bbox_inches, **kwargs)
#     fig.savefig(f"{path}.svg", dpi=400, bbox_inches=bbox_inches, **kwargs)
#     if tiff:
#         fig.savefig(f"{path}.tiff", dpi=400, bbox_inches=bbox_inches, **kwargs)
#     # plt.close(fig)

import matplotlib.pyplot as plt
from contextlib import contextmanager


@contextmanager
def scale_font(scale_factor):
    # 保存当前字体设置的副本
    original_font_params = {
        "font.size": plt.rcParams["font.size"],
        "axes.titlesize": plt.rcParams["axes.titlesize"],
        "axes.labelsize": plt.rcParams["axes.labelsize"],
        "xtick.labelsize": plt.rcParams["xtick.labelsize"],
        "ytick.labelsize": plt.rcParams["ytick.labelsize"],
        "legend.fontsize": plt.rcParams["legend.fontsize"],
    }

    try:
        # 调整字体大小
        plt.rcParams["font.size"] = plt.rcParams["font.size"] * scale_factor
        plt.rcParams["axes.titlesize"] = plt.rcParams["axes.titlesize"] * scale_factor
        plt.rcParams["axes.labelsize"] = plt.rcParams["axes.labelsize"] * scale_factor
        plt.rcParams["xtick.labelsize"] = plt.rcParams["xtick.labelsize"] * scale_factor
        plt.rcParams["ytick.labelsize"] = plt.rcParams["ytick.labelsize"] * scale_factor
        plt.rcParams["legend.fontsize"] = plt.rcParams["legend.fontsize"] * scale_factor

        yield  # 在此暂停并执行with语句内的代码
    finally:
        # 恢复原始字体设置
        plt.rcParams.update(original_font_params)


def save_fig(
    fig=None,
    path=None,
    bbox_inches="tight",
    dpi=400,
    tiff=False,
    tiff_compress=False,
    **kwargs,
):
    if path is None:
        path = "temp"
    if fig is None or isinstance(fig, plt.Figure):
        if fig is None:
            fig = plt.gcf()
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(f"{path}.png", dpi=dpi, bbox_inches=bbox_inches, **kwargs)
        fig.savefig(f"{path}.pdf", dpi=dpi, bbox_inches=bbox_inches, **kwargs)
        fig.savefig(f"{path}.svg", dpi=dpi, bbox_inches=bbox_inches, **kwargs)
        if tiff:
            fig.savefig(f"{path}.tiff", dpi=dpi, bbox_inches=bbox_inches, **kwargs)
        # plt.close(fig)
    elif isinstance(fig, ggplot):
        fig.save(f"{path}.png", dpi=dpi, **kwargs)
        fig.save(f"{path}.pdf", dpi=dpi, **kwargs)
        fig.save(f"{path}.svg", dpi=dpi, **kwargs)
        if tiff:
            if tiff_compress:
                fig.save(
                    f"{path}.tiff",
                    dpi=dpi,
                    pil_kwargs={"compression": "tiff_lzw"},
                    **kwargs,
                )
            else:
                fig.save(f"{path}.tiff", dpi=dpi, **kwargs)
