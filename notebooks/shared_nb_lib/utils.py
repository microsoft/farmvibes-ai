import os
from typing import List, Tuple

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from vibe_core.data.core_types import gen_guid


def create_geojson_file_from_point(
    list_of_points: List[Point], labels: List[int], prompt_ids: List[int], storage_dirpath: str
) -> Tuple[str, gpd.GeoDataFrame, str]:
    """
    Create a geojson file from a list of points, labels, and prompt_ids
    """
    file_name_prefix = gen_guid()
    df = pd.DataFrame({"geometry": list_of_points, "label": labels, "prompt_id": prompt_ids})

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")  # type: ignore

    gdf.to_file(
        os.path.join(storage_dirpath, f"{file_name_prefix}_geometry_collection.geojson"),
        driver="GeoJSON",
    )

    op_points_filepath = f"/mnt/{file_name_prefix}_geometry_collection.geojson"
    return op_points_filepath, gdf, file_name_prefix
