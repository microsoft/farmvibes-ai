# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
from shapely.geometry import Polygon, mapping

from vibe_core.data import Raster
from vibe_core.data.products import BingMapsProduct
from vibe_dev.testing.op_tester import OpTester
from vibe_lib.bing_maps import BingMapsCollection

FAKE_GEOMETRY = Polygon(
    [
        (46.998848, -118.940490),
        (46.998848, -118.876148),
        (47.013422, -118.876148),
        (47.013422, -118.940490),
    ]
)
FAKE_TIME_RANGE = (datetime.now(), datetime.now())


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_bing_basemap.yaml")


def create_blank_jpeg(_: str, out_path: str):
    data = np.zeros((100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(data)
    img.save(out_path)


@patch.object(
    BingMapsCollection,
    "download_tile",
    side_effect=create_blank_jpeg,
)
@patch.object(
    BingMapsCollection,
    "get_download_url_and_subdomains",
    return_value=("fake_download_url_{subdomain}_{quadkey}_{api_key}", ["fake_subdomain"]),
)
def test_op(_: MagicMock, __: MagicMock):
    input_product = BingMapsProduct(
        id="fake_product",
        time_range=FAKE_TIME_RANGE,
        geometry=mapping(FAKE_GEOMETRY),  # type: ignore
        assets=[],
        url="fake_url",
        zoom_level=1,
        imagery_set="Aerial",
        map_layer="Basemap",
        orientation=0.0,
    )

    op_tester = OpTester(CONFIG_PATH)
    op_tester.update_parameters({"api_key": "fake_api_key"})
    output_data = op_tester.run(**{"input_product": input_product})

    # Get op result
    output_name = "basemap"
    assert output_name in output_data
    output_basemap = output_data[output_name]
    assert isinstance(output_basemap, Raster)
    assert len(output_basemap.assets) == 1
