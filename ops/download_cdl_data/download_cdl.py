# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import mimetypes
import os
from tempfile import TemporaryDirectory
from typing import Any, Dict
from zipfile import ZipFile

import numpy as np
import pandas as pd

from vibe_core.data import AssetVibe, CategoricalRaster, gen_guid
from vibe_core.data.products import CDL_DOWNLOAD_URL, CDLProduct
from vibe_core.file_downloader import download_file
from vibe_lib.raster import (
    INT_COMPRESSION_KWARGS,
    compress_raster,
    json_to_asset,
    step_cmap_from_colors,
)


def download_cdl_tif(cdl_product: CDLProduct, out_path: str) -> None:
    """Download the CDL zip and decompress the .tif file and recompress it to out_path"""
    cdl_year = cdl_product.time_range[0].year

    with TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, f"cdl_{cdl_year}.zip")
        product_url = CDL_DOWNLOAD_URL.format(cdl_year)
        download_file(product_url, zip_path)

        with ZipFile(zip_path) as zf:
            zip_member = [f for f in zf.filelist if f.filename.endswith(".tif")][0]
            # Trick to extract file without the whole directory tree
            # https://stackoverflow.com/questions/4917284/
            zip_member.filename = os.path.basename(zip_member.filename)
            file_path = zf.extract(zip_member, path=tmp)
            compress_raster(file_path, out_path, **INT_COMPRESSION_KWARGS)


class CallbackBuilder:
    MIN_CLASS_IDX: int = 0
    MAX_CLASS_IDX: int = 255

    def __init__(self, metadata_path: str):
        self.tmp_dir = TemporaryDirectory()
        self.df = pd.read_excel(metadata_path, header=3, index_col=0).dropna(axis=1)
        cmap = self.df[["Erdas_Red", "Erdas_Green", "Erdas_Blue"]].values.astype(float)
        # Add alpha value
        self.cmap = np.concatenate((cmap, cmap.sum(axis=1)[:, None] > 0), axis=1)

    def __call__(self):
        def cdl_callback(input_product: CDLProduct) -> Dict[str, CategoricalRaster]:
            """
            This op receives a CDLProduct (probably from list_cdl_products op) and
            downloads the zipped CDL map. It decompress the .tif file from it and yields
            a CategoricalRaster with references to that asset
            """

            out_id = gen_guid()
            filepath = os.path.join(self.tmp_dir.name, f"{out_id}.tif")

            download_cdl_tif(input_product, filepath)

            new_asset = AssetVibe(reference=filepath, type=mimetypes.types_map[".tif"], id=out_id)

            vis_dict: Dict[str, Any] = {
                "bands": [0],
                "colormap": step_cmap_from_colors(
                    self.cmap, range(self.MIN_CLASS_IDX + 1, self.MAX_CLASS_IDX + 1)
                ),
                "range": (self.MIN_CLASS_IDX, self.MAX_CLASS_IDX),
            }

            raster = CategoricalRaster.clone_from(
                input_product,
                id=gen_guid(),
                assets=[new_asset, json_to_asset(vis_dict, self.tmp_dir.name)],
                bands={"categories": 0},
                categories=self.df["Class_Names"].tolist(),
            )

            return {"cdl_raster": raster}

        return cdl_callback

    def __del__(self):
        self.tmp_dir.cleanup()
