import io
from datetime import datetime

import matplotlib.pyplot as plt
from IPython.display import Image, display
from shapely import geometry as shpg
from torchgeo.datasets import BoundingBox


def bbox_to_shapely(bbox: BoundingBox) -> shpg.Polygon:
    """
    Convert from torchgeo's BoundingBox to a shapely polygon
    """
    return shpg.box(bbox.minx, bbox.miny, bbox.maxx, bbox.maxy)


def format_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y/%m/%d")


def lw_plot():
    """
    Compress images to make notebook smaller
    """
    iobytes = io.BytesIO()
    plt.savefig(iobytes, format="jpg", bbox_inches="tight")
    plt.close()
    iobytes.seek(0)
    display(Image(data=iobytes.read()))
