from datetime import datetime
from typing import Any, Dict, List, NamedTuple, Sequence, cast

import geopandas as gpd
import pandas as pd
import rasterio
from pandas.core.frame import DataFrame
from rasterstats import zonal_stats
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry


class Stats(NamedTuple):
    date: datetime
    min: float
    max: float
    mean: float


def calculate_zonal_stats(
    raster_paths: Sequence[str], raster_dates: Sequence[datetime], geo_dict: Dict[str, Any]
) -> List[Stats]:
    """For each raster in a list of rasters, calculates min, max, and mean
    values of the pixels overlapping or intersecting a geojson geometry.
    This function assumes geometry represents a single non multi geometry.
    """

    # Convert geometry to raster CRS
    with rasterio.open(raster_paths[0]) as src:  # type: ignore
        crs = src.crs  # type: ignore
    geom: BaseGeometry = (
        gpd.GeoSeries(shape(geo_dict), crs="epsg:4326").to_crs(crs).iloc[0]  # type: ignore
    )

    result: List[Stats] = []

    for raster_path, raster_date in zip(raster_paths, raster_dates):
        stats = zonal_stats(geom, raster_path)

        raster_stats = Stats(
            raster_date,
            cast(float, stats[0]["min"]),
            cast(float, stats[0]["max"]),
            cast(float, stats[0]["mean"]),
        )

        result.append(raster_stats)

    return result


def convert_zonal_stats_to_timeseries(stats: Sequence[Stats]) -> DataFrame:
    df = pd.DataFrame(stats)
    df.set_index("date", drop=True, inplace=True)  # type: ignore

    return df
