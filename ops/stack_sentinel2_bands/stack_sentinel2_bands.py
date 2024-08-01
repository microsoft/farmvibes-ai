# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import mimetypes
import os
from tempfile import TemporaryDirectory
from typing import Dict, List, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling

from vibe_core.data import (
    AssetVibe,
    DownloadedSentinel2Product,
    Sentinel2CloudMask,
    Sentinel2Raster,
    gen_guid,
)
from vibe_lib.raster import INT_COMPRESSION_KWARGS, open_raster_from_ref

BAND_ORDER: List[str] = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
]

CLOUD_CATEGORIES = ["NO-CLOUD", "OPAQUE", "CIRRUS", "OTHER"]
LOGGER = logging.getLogger(__name__)


def save_stacked_raster(band_filepaths: Sequence[str], ref_filepath: str, out_path: str) -> None:
    """
    Save raster by stacking all bands.
    Reprojects all bands to match the reference band file provided
    """
    with open_raster_from_ref(ref_filepath) as src:
        meta = src.meta
    out_meta = meta.copy()
    out_meta.update(
        {
            "count": len(band_filepaths),
            "driver": "GTiff",
            "nodata": 0,
            **INT_COMPRESSION_KWARGS,
        }
    )

    vrt_options = {
        "resampling": Resampling.bilinear,
        "crs": meta["crs"],
        "transform": meta["transform"],
        "height": meta["height"],
        "width": meta["width"],
    }

    with open_raster_from_ref(out_path, "w", **out_meta) as dst:
        for i, path in enumerate(band_filepaths):
            with open_raster_from_ref(path) as src:
                with WarpedVRT(src, **vrt_options) as vrt:
                    data = vrt.read(1)
            dst.write(data, i + 1)


def rasterize_clouds(item: DownloadedSentinel2Product, ref_file: str, out_path: str) -> None:
    """
    Rasterize cloud shapes and save compressed tiff file.
    """
    with open_raster_from_ref(ref_file) as src:
        meta = src.meta
    meta.update({"nodata": 100, "driver": "GTiff", "dtype": "uint8", **INT_COMPRESSION_KWARGS})
    out = np.zeros((meta["height"], meta["width"]))
    try:
        gml_path = item.get_downloaded_cloudmask().path_or_url
        df = gpd.read_file(gml_path, WRITE_GFS="NO")
        cloud_map = {
            "OPAQUE": CLOUD_CATEGORIES.index("OPAQUE"),
            "CIRRUS": CLOUD_CATEGORIES.index("CIRRUS"),
        }
        values = (
            df["maskType"].map(cloud_map).fillna(CLOUD_CATEGORIES.index("OTHER"))  # type: ignore
        )
        rasterize(
            ((g, v) for g, v in zip(df["geometry"], values)),  # type: ignore
            out=out,
            transform=meta["transform"],
        )
    except ValueError:
        # Empty file means no clouds
        LOGGER.debug(
            "ValueError when opening cloud GML file. Assuming there are no clouds and ignoring.",
            exc_info=True,
        )
        pass
    except KeyError:
        LOGGER.warning(f"No cloudmask available on downloaded product {item.product_name}")
    with open_raster_from_ref(out_path, "w", **meta) as dst:
        dst.write(out, 1)


def process_s2(
    item: DownloadedSentinel2Product, output_file_name: str, tmp_folder: str
) -> Tuple[str, str, List[str]]:
    output_img_path = os.path.join(tmp_folder, output_file_name)
    output_cloud_path = os.path.join(tmp_folder, "cloudmask.tif")

    # Make sure bands are in order
    valid_bands = [b for b in BAND_ORDER if b in item.asset_map]
    band_filepaths = [item.get_downloaded_band(b).path_or_url for b in valid_bands]
    ref_filepath = band_filepaths[BAND_ORDER.index("B02")]
    save_stacked_raster(band_filepaths, ref_filepath, output_img_path)

    # Generate cloud mask
    rasterize_clouds(item, ref_filepath, output_cloud_path)

    return output_img_path, output_cloud_path, valid_bands


class CallbackBuilder:
    def __init__(self):
        self.tmp_dir = TemporaryDirectory()

    def __call__(self):
        def process_sentinel_2(
            input_item: DownloadedSentinel2Product,
        ) -> Dict[str, Union[Sentinel2Raster, Sentinel2CloudMask]]:
            ref_name: str = input_item.product_name
            output_file_name = ref_name + ".tif"
            tmp_dir = os.path.join(self.tmp_dir.name, ref_name)
            os.makedirs(tmp_dir)

            img, cloud, valid_bands = process_s2(input_item, output_file_name, tmp_dir)

            img_asset = AssetVibe(reference=img, type=mimetypes.types_map[".tif"], id=gen_guid())
            cloud_asset = AssetVibe(
                reference=cloud, type=mimetypes.types_map[".tif"], id=gen_guid()
            )

            bands = Sentinel2Raster.clone_from(
                input_item,
                bands={name: idx for idx, name in enumerate(valid_bands)},
                id=ref_name,
                assets=[img_asset],
            )

            cloud = Sentinel2CloudMask.clone_from(
                input_item,
                bands={"cloud": 0},
                categories=CLOUD_CATEGORIES,
                id=ref_name,
                assets=[cloud_asset],
            )

            return {"sentinel2_raster": bands, "sentinel2_cloud_mask": cloud}

        return process_sentinel_2

    def __del__(self):
        self.tmp_dir.cleanup()
