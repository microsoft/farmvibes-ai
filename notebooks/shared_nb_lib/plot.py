import io
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from IPython.core.display import Image
from IPython.display import display
from matplotlib.colors import ListedColormap


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
