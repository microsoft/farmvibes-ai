import io
from copy import deepcopy
from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.display import Image
from IPython.display import display
from matplotlib.colors import ListedColormap
from numpy._typing import NDArray


def lw_plot():
    """
    Compress images to make notebook smaller by using lossy (JPEG) compression
    """
    iobytes = io.BytesIO()
    plt.savefig(iobytes, format="jpg", bbox_inches="tight")
    plt.close()
    iobytes.seek(0)
    display(Image(data=iobytes.read()))


def transparent_cmap(cmap: ListedColormap, max_alpha: float = 0.8, N: int = 255):
    "Copy colormap and set alpha values"
    mycmap = deepcopy(cmap)
    mycmap._init()  # type: ignore
    mycmap._lut[:, -1] = np.linspace(0, max_alpha, N + 4)  # type: ignore
    return mycmap


def plot_categorical_map(
    dataset: List[List[float]],
    color_dict: Dict[int, str],
    labels: List[str],
    geom_exterior: Optional[NDArray[Any]] = None,
    extent: Optional[List[float]] = None,
    title: str = "Category Map",
    xlabel: str = "longitude",
    ylabel: str = "latitude",
):
    # Create a colormap from the color dictionary
    cmap = ListedColormap([color_dict[x] for x in color_dict.keys()])  # type: ignore

    # Prepare normalizer
    norm_bins = np.sort([*color_dict.keys()]) + 0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    norm = matplotlib.colors.BoundaryNorm(norm_bins, len(labels), clip=True)
    fmt = matplotlib.ticker.FuncFormatter(lambda x, _: labels[norm(x)])  # type: ignore

    # Plot the figure
    fig, ax = plt.subplots()

    extent = extent or [0, len(dataset[0]), 0, len(dataset)]

    im = ax.imshow(dataset, cmap=cmap, extent=extent, norm=norm)

    if geom_exterior is not None:
        # Plot geom on top of the cropped image
        plt.plot(*geom_exterior, color="red")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2
    fig.colorbar(im, format=fmt, ticks=tickz)
    plt.show()
