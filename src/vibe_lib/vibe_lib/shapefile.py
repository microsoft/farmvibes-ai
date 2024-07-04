import os
from typing import Any

import geopandas as gpd
import numpy as np
from numpy.typing import NDArray
from rasterio import Affine
from rasterio.crs import CRS
from rasterio.features import shapes
from shapely import geometry as shpg

from vibe_core.data.core_types import AssetVibe, gen_guid
from vibe_lib.archive import create_flat_archive
from vibe_lib.geometry import SimplifyBy


def write_shapefile(
    data: NDArray[Any],
    input_crs: CRS,
    tr: Affine,
    mask1: NDArray[Any],
    path: str,
    simplify: str,
    tolerance: float,
    file_name: str,
    output_crs: int = 4326,
) -> AssetVibe:
    clusters = np.unique(data)
    data1 = data * mask1.astype(np.uint16)

    for segment in clusters:
        cluster = data1 == segment
        df_shapes = gpd.GeoSeries(
            [shpg.shape(s) for s, _ in shapes(data1.astype(np.uint16), mask=cluster, transform=tr)],
            crs=input_crs,
        )  # type: ignore
        cluster_path = os.path.join(path, f"{file_name}{segment}.shp")

        if simplify == SimplifyBy.simplify:
            df_shapes.simplify(tolerance).to_crs(output_crs).to_file(cluster_path)
        elif simplify == SimplifyBy.convex:
            df_shapes.convex_hull.to_file(cluster_path)
        else:
            df_shapes.to_file(cluster_path)

    # Create zip archive containing all output
    archive_path = create_flat_archive(path, "result")
    return AssetVibe(reference=archive_path, type="application/zip", id=gen_guid())
