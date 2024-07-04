import mimetypes
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Tuple, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import shapely.geometry as shpg
from rasterio.merge import merge
from shapely.geometry.base import BaseGeometry

from vibe_core.data import AssetVibe, CategoricalRaster, DataVibe, gen_guid
from vibe_lib.raster import json_to_asset, step_cmap_from_colors

SERVICE_URL = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile"
CDL_CRS = "epsg:5070"
# Maximum area per request is 2M square km, 2e11 seems to work better
MAX_AREA = 1e11


def download_file(url: str, out_path: str) -> None:
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def split_geometry(geom: BaseGeometry, max_area: float) -> List[BaseGeometry]:
    if geom.area < max_area:
        # Done
        return [geom]

    # Split it!
    x0, y0, x1, y1 = cast(Tuple[int, int, int, int], geom.bounds)
    if (x1 - x0) > (y1 - y0):
        # Split along width
        b1 = shpg.box(x0, y0, (x0 + x1) / 2, y1)
        b2 = shpg.box((x0 + x1) / 2, y0, x1, y1)
    else:
        # Split along height
        b1 = shpg.box(x0, y0, x1, (y0 + y1) / 2)
        b2 = shpg.box(x0, (y0 + y1) / 2, x1, y1)
    return split_geometry(b1, max_area) + split_geometry(b2, max_area)


def get_cdl_url(geom: BaseGeometry, dt: datetime) -> str:
    formatted_bbox = ",".join([f"{b:.1f}" for b in geom.bounds])
    payload = {"year": str(dt.year), "bbox": formatted_bbox}
    r = requests.get(SERVICE_URL, params=payload)
    r.raise_for_status()
    e = ET.fromstring(r.text)
    tif_url = list(e)[0].text
    if tif_url is None:
        raise ValueError(f"URL is missing from response {r.text}")
    return tif_url


def save_cdl_tif(geom: BaseGeometry, dt: datetime, out_path: str) -> None:
    split_geoms = [g for g in split_geometry(geom, MAX_AREA) if g.intersects(geom)]
    with TemporaryDirectory() as tmp:
        split_paths = [os.path.join(tmp, f"{i}.tif") for i in range(len(split_geoms))]
        for g, p in zip(split_geoms, split_paths):
            tif_url = get_cdl_url(g, dt)
            download_file(tif_url, p)
        if len(split_geoms) > 1:
            # Merge all parts into a single tiff
            merge(split_paths, bounds=geom.bounds, dst_path=out_path)
        else:
            os.rename(split_paths[0], out_path)


class CallbackBuilder:
    MIN_CLASS_IDX: int = 0
    MAX_CLASS_IDX: int = 255

    def __init__(self, metadata_url: str):
        self.tmp_dir = TemporaryDirectory()
        self.df = pd.read_excel(metadata_url, header=3, index_col=0).dropna(axis=1)
        cmap = self.df[["Erdas_Red", "Erdas_Green", "Erdas_Blue"]].values.astype(float)
        # Add alpha value
        self.cmap = np.concatenate((cmap, cmap.sum(axis=1)[:, None] > 0), axis=1)

    def __call__(self):
        def cdl_callback(input_data: DataVibe) -> CategoricalRaster:
            proj_geom: BaseGeometry = (
                gpd.GeoSeries(shpg.shape(input_data.geometry), crs="epsg:4326")
                .to_crs(CDL_CRS)
                .iloc[0]
            )
            # We are taking the year in the middle point of the time range for now
            dt = datetime.fromtimestamp(sum(d.timestamp() for d in input_data.time_range) / 2)
            out_id = gen_guid()
            filepath = os.path.join(self.tmp_dir.name, f"{out_id}.tif")
            save_cdl_tif(proj_geom, dt, filepath)
            new_asset = AssetVibe(reference=filepath, type=mimetypes.types_map[".tif"], id=out_id)

            vis_dict: Dict[str, Any] = {
                "bands": [0],
                "colormap": step_cmap_from_colors(
                    self.cmap, range(self.MIN_CLASS_IDX + 1, self.MAX_CLASS_IDX + 1)
                ),
                "range": (self.MIN_CLASS_IDX, self.MAX_CLASS_IDX),
            }

            raster = CategoricalRaster.clone_from(
                input_data,
                id=gen_guid(),
                assets=[new_asset, json_to_asset(vis_dict, self.tmp_dir.name)],
                bands={"categories": 0},
                categories=self.df["Class_Names"].tolist(),
            )

            return raster

        def cdl_callback_list(input_data: List[DataVibe]) -> Dict[str, List[CategoricalRaster]]:
            return {"cdl_rasters": [cdl_callback(input_datum) for input_datum in input_data]}

        return cdl_callback_list

    def __del__(self):
        self.tmp_dir.cleanup()
