# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import os
from collections import defaultdict
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, cast

import geopandas as gpd
import h5py
import numpy as np
from geopandas.array import GeometryArray
from shapely import geometry as shpg
from shapely.geometry.base import BaseGeometry

from vibe_core.data import AssetVibe, DataVibe, GEDIProduct, GeometryCollection, gen_guid
from vibe_core.data.core_types import BBox

BEAMS = [
    "BEAM0000",
    "BEAM0001",
    "BEAM0010",
    "BEAM0011",
    "BEAM0101",
    "BEAM0110",
    "BEAM1000",
    "BEAM1011",
]
L2B = "GEDI02_B.002"
LOGGER = logging.getLogger(__name__)


def extract_dataset(filepath: str, geometry: BaseGeometry, check_quality: bool):
    lon_min, lat_min, lon_max, lat_max = cast(BBox, geometry.bounds)
    d: Dict[str, List[Any]] = defaultdict(list)
    with h5py.File(filepath) as h5:
        for b in BEAMS:
            lon = cast(h5py.Dataset, h5.get(f"{b}/geolocation/lon_lowestmode"))[()]
            lat = cast(h5py.Dataset, h5.get(f"{b}/geolocation/lat_lowestmode"))[()]
            bbox_mask = (lon_min <= lon) & (lon <= lon_max) & (lat_min <= lat) & (lat <= lat_max)
            if not bbox_mask.any():
                continue
            bbox_idx = np.where(bbox_mask)[0]
            pts = gpd.points_from_xy(lon[bbox_idx], lat[bbox_idx])
            within = pts.within(geometry)
            if not within.any():
                continue
            within_idx = np.where(within)[0]
            idx = bbox_idx[within_idx]

            if check_quality:
                # Filter data by quality flag: 1 = good, 0 = bad
                qual = cast(h5py.Dataset, h5.get(f"{b}/l2b_quality_flag"))[idx].astype(bool)
                if not qual.any():
                    continue
                within_idx = within_idx[qual]
                idx = idx[qual]

            d["geometry"].extend(cast(GeometryArray, pts[within_idx]))
            d["beam"].extend(cast(h5py.Dataset, h5.get(f"{b}/beam"))[idx])
            d["rh100"].extend(cast(h5py.Dataset, h5.get(f"{b}/rh100"))[idx])
    if not d or any(not v for v in d.values()):
        return None
    df = gpd.GeoDataFrame(d, crs="epsg:4326")  # type: ignore
    return df


class CallbackBuilder:
    def __init__(self, check_quality: bool):
        self.tmp_dir = TemporaryDirectory()
        self.check_quality = check_quality

    def __call__(self):
        def callback(gedi_product: GEDIProduct, roi: DataVibe) -> Dict[str, GeometryCollection]:
            if gedi_product.processing_level != L2B:
                raise ValueError(
                    f"Processing level must be {L2B}, found {gedi_product.processing_level}"
                )
            h5_path = gedi_product.assets[0].local_path
            geom = shpg.shape(roi.geometry)
            asset_guid = gen_guid()
            LOGGER.info(f"Extracting data from hdf5 file {h5_path}")
            df = extract_dataset(h5_path, geom, self.check_quality)
            if df is not None:
                asset_path = os.path.join(self.tmp_dir.name, f"{asset_guid}.gpkg")
                LOGGER.info(f"Saving data to {asset_path}")
                df.to_file(asset_path, driver="GPKG")
                LOGGER.info("All done! Creating GeometryCollection")

                assets = [
                    AssetVibe(
                        reference=asset_path, type="application/geopackage+sqlite3", id=asset_guid
                    )
                ]
            else:
                LOGGER.info(
                    f"No data available in product {gedi_product.product_name} after filtering, "
                    "creating assetless output"
                )
                assets = []
            rh100 = GeometryCollection.clone_from(
                gedi_product, geometry=roi.geometry, id=gen_guid(), assets=assets
            )
            return {"rh100": rh100}

        return callback
