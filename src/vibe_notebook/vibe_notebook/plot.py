# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Auxiliary methods for plotting and visualizing data in notebooks."""

import io
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.display import Image
from IPython.display import display
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from numpy._typing import NDArray


def lw_plot():
    """Compress images to make notebook smaller by using lossy (JPEG) compression."""
    iobytes = io.BytesIO()
    plt.savefig(iobytes, format="jpg", bbox_inches="tight")
    plt.close()
    iobytes.seek(0)
    display(Image(data=iobytes.read()))


def transparent_cmap(cmap: ListedColormap, max_alpha: float = 0.8, N: int = 255) -> ListedColormap:
    """Define a transparent colormap based on an input colormap."""
    mycmap = deepcopy(cmap)
    mycmap._init()  # type: ignore
    mycmap._lut[:, -1] = np.linspace(0, max_alpha, N + 4)  # type: ignore
    return mycmap


def _plot_categorical_map(
    dataset: List[List[float]],
    color_dict: Dict[int, str],
    labels: List[str],
    geom_exterior: Optional[NDArray[Any]] = None,
    extent: Optional[List[float]] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
):
    # Plot the figure
    if not fig or not ax:
        fig, ax = plt.subplots()

    # Create a colormap from the color dictionary
    cmap = ListedColormap([color_dict[x] for x in color_dict.keys()])  # type: ignore

    # Prepare normalizer
    norm_bins = np.sort([*color_dict.keys()]) + 0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    norm = matplotlib.colors.BoundaryNorm(norm_bins, len(labels), clip=True)  # type: ignore
    fmt = matplotlib.ticker.FuncFormatter(lambda x, _: labels[norm(x)])  # type: ignore

    extent = extent or [0, len(dataset[0]), 0, len(dataset)]

    im = ax.imshow(dataset, cmap=cmap, extent=extent, norm=norm)  # type: ignore

    if geom_exterior is not None:
        # Plot geom on top of the cropped image
        ax.plot(*geom_exterior, color="red")  # type: ignore

    if title:
        ax.set_title(title)  # type: ignore
    if xlabel:
        ax.set_xlabel(xlabel)  # type: ignore
    if ylabel:
        ax.set_ylabel(ylabel)  # type: ignore

    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2

    return im, fmt, tickz


def plot_categorical_map(
    dataset: List[List[float]],
    color_dict: Dict[int, str],
    labels: List[str],
    geom_exterior: Optional[NDArray[Any]] = None,
    extent: Optional[List[float]] = None,
    title: str = "Category Map",
    xlabel: str = "longitude",
    ylabel: str = "latitude",
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
):
    """Plot a categorical map with a color dictionary."""
    im, fmt, tickz = _plot_categorical_map(
        dataset=dataset,
        color_dict=color_dict,
        labels=labels,
        geom_exterior=geom_exterior,
        extent=extent,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        fig=fig,
        ax=ax,
    )

    plt.colorbar(im, format=fmt, ticks=tickz)
    plt.show()

    return im, fmt, tickz


def plot_categorical_maps(
    datasets: List[List[List[float]]],
    color_dict: Dict[int, str],
    labels: List[str],
    titles: List[str],
    suptitle: str,
    geom_exterior: Optional[NDArray[Any]] = None,
    extent: Optional[List[float]] = None,
    xlabel: str = "",
    ylabel: str = "",
    n_cols: int = 2,
    figsize: Tuple[int, int] = (12, 10),
):
    """Plot multiple categorical maps side by side."""
    rows = int(np.ceil(len(datasets) / n_cols))
    fig, axes = plt.subplots(rows, n_cols, figsize=figsize, sharex=True, sharey=True)

    im, fmt, tickz = None, None, None
    for i, dataset in enumerate(datasets):
        im, fmt, tickz = _plot_categorical_map(
            dataset=dataset,
            color_dict=color_dict,
            labels=labels,
            geom_exterior=geom_exterior,
            extent=extent,
            title=titles[i],
            fig=fig,
            ax=axes[i // n_cols, i % n_cols],  # type: ignore
        )
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.suptitle(suptitle)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])  # type: ignore
    fig.colorbar(im, cax=cbar_ax, format=fmt, ticks=tickz)  # type: ignore

    plt.show()
