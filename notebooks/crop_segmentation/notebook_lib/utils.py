from datetime import datetime

from shapely import geometry as shpg
from torchgeo.datasets import BoundingBox


def bbox_to_shapely(bbox: BoundingBox) -> shpg.Polygon:
    """
    Convert from torchgeo's BoundingBox to a shapely polygon
    """
    return shpg.box(bbox.minx, bbox.miny, bbox.maxx, bbox.maxy)


def format_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y/%m/%d")
except ValueError as e:
        # Handle potential errors with conversion (e.g., invalid timestamp)
        return "Invalid timestamp"
