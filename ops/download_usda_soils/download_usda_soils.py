import os
import zipfile
from datetime import datetime
from tempfile import TemporaryDirectory
from typing import Dict, Optional

import pandas as pd
import rasterio
from shapely import geometry as shpg
from shapely.geometry import mapping

from vibe_core.data import AssetVibe, CategoricalRaster, DataVibe
from vibe_core.data.core_types import gen_guid, gen_hash_id
from vibe_core.file_downloader import download_file
from vibe_lib.raster import json_to_asset


class CallbackBuilder:
    def __init__(self, url: str, zip_file: str, tiff_file: str, meta_file: str):
        self.url = url
        self.zip_file = zip_file
        self.tiff_file = tiff_file
        self.meta_file = meta_file
        self.tmp_dir = TemporaryDirectory()

    def __call__(self):
        def download_usda_soils(
            input_item: DataVibe,
        ) -> Dict[str, Optional[CategoricalRaster]]:
            fpath = os.path.join(self.tmp_dir.name, self.zip_file)
            ftiff = os.path.join(self.tmp_dir.name, self.tiff_file)
            fmeta = os.path.join(self.tmp_dir.name, self.meta_file)

            download_file(self.url, fpath)
            with zipfile.ZipFile(fpath) as zf:
                with open(ftiff, "wb") as f:
                    f.write(zf.read(self.tiff_file))
                with open(fmeta, "wb") as f:
                    f.write(zf.read(self.meta_file))

            vibe_asset = AssetVibe(reference=ftiff, type="image/tiff", id=gen_guid())

            with rasterio.open(ftiff) as ds:
                geometry = mapping(shpg.box(*ds.bounds))

            classes = pd.read_table(fmeta, index_col=0)
            classes = classes["SOIL_ORDER"] + ":" + classes["SUBORDER"]  # type: ignore
            classes = {v: k for k, v in classes.to_dict().items()}

            downloaded_raster = CategoricalRaster.clone_from(
                input_item,
                id=gen_hash_id(
                    "usda_soil",
                    geometry,
                    (datetime(2015, 1, 1), datetime(2015, 12, 31)),  # dummy dates
                ),
                assets=[vibe_asset, json_to_asset(classes, self.tmp_dir.name)],
                time_range=input_item.time_range,
                geometry=geometry,
                bands={"soil_order:suborder": 0},
                categories=list(classes.keys()),
            )
            return {"downloaded_raster": downloaded_raster}

        return download_usda_soils

    def __del__(self):
        self.tmp_dir.cleanup()
