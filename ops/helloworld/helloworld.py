# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from tempfile import TemporaryDirectory
from typing import Dict, List, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely import geometry as shpg

from vibe_core.data import DataVibe, Raster, gen_guid
from vibe_core.data.core_types import AssetVibe
from vibe_lib.raster import INT_COMPRESSION_KWARGS

FONT_PATHS = [
    "DejaVuSans.ttf",
    "/opt/conda/fonts/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
]


def load_default_font():
    font_set = False
    for font in FONT_PATHS:
        try:
            ImageDraw.ImageDraw.font = ImageFont.truetype(font, 14)  # type: ignore
            font_set = True
            break
        except OSError:
            pass
    if not font_set:
        # We failed to load the font, raise an error
        raise ValueError("Failed to load font for helloworld op")


def get_geoms(g: Union[shpg.Polygon, shpg.MultiPolygon]) -> List[shpg.Polygon]:
    """
    Map MultiPolygons and Polygons into list of Polygons
    """
    if isinstance(g, shpg.MultiPolygon):
        return list(g.geoms)
    return [g]


class CallbackBuilder:
    msg = "HELLO WORLD"

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.tmp_dir = TemporaryDirectory()
        load_default_font()

    def __call__(self):
        def hello(user_input: DataVibe) -> Dict[str, Raster]:
            geom = shpg.shape(user_input.geometry)
            df = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))  # type: ignore
            # Find out which geometries intersect with the input geometry
            # Some countries have several polygons, let's split MultiPolygons into Polygons
            # So we don't have regions that are far away being highlighted
            country_geoms = [gg for g in df.geometry.tolist() for gg in get_geoms(g)]
            yes_geom = [(g, 1) for g in country_geoms if g.intersects(geom)]
            no_geom = [(g, 2) for g in country_geoms if not g.intersects(geom)]
            tr = from_bounds(-180, -90, 180, 90, self.width, self.height)
            # Generate RGBA image using tab10 (blue, orange, and green)
            ar = (
                plt.cm.tab10(  # type: ignore
                    rasterize(
                        yes_geom + no_geom + [(geom.boundary, 3)],
                        out_shape=(self.height, self.width),
                        transform=tr,  # type: ignore
                    )
                )
                * 255
            ).astype(np.uint8)

            # Let's write a nice message 🙂
            img = Image.fromarray(ar)
            img_d = ImageDraw.Draw(img)
            offset = (self.width - img_d.getfont().getbbox(self.msg)[3]) // 2
            img_d.text((offset, 10), "HELLO WORLD", fill=(255, 255, 255))
            # Get image into CHW array and pick RGB bands
            ar = np.array(img).transpose((2, 0, 1))[:3]

            # Write image to tiff file with the correct CRS and transform
            meta = {
                "driver": "GTiff",
                "dtype": "uint8",
                "width": self.width,
                "height": self.height,
                "count": 3,
                "crs": "epsg:4326",
                "transform": tr,
            }
            raster_guid = gen_guid()
            out_path = os.path.join(self.tmp_dir.name, f"{raster_guid}.tif")
            with rasterio.open(out_path, "w", **meta, **INT_COMPRESSION_KWARGS) as dst:
                dst.write(ar)
            asset = AssetVibe(out_path, "image/tiff", raster_guid)
            # Let's use the geometry and date from the input
            return {
                "raster": Raster.clone_from(
                    user_input,
                    id=gen_guid(),
                    assets=[asset],
                    bands={"red": 0, "blue": 1, "green": 2},
                )
            }

        return hello

    def __del__(self):
        self.tmp_dir.cleanup()
