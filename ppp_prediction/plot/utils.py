import matplotlib.pyplot as plt
from pathlib import Path


def save_fig(fig=None, path=None, bbox_inches="tight", dpi=400, tiff=False, **kwargs):
    if path is None:
        path = "temp"
    if fig is None:
        fig = plt.gcf()
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(f"{path}.png", dpi=400, bbox_inches=bbox_inches, **kwargs)
    fig.savefig(f"{path}.pdf", dpi=400, bbox_inches=bbox_inches, **kwargs)
    fig.savefig(f"{path}.svg", dpi=400, bbox_inches=bbox_inches, **kwargs)
    if tiff:
        fig.savefig(f"{path}.tiff", dpi=400, bbox_inches=bbox_inches, **kwargs)
    plt.close(fig)
