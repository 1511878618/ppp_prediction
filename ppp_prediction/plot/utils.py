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
