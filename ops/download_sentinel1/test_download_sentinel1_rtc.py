import os
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import planetary_computer as pc
import pytest
import rasterio
from pystac import Asset, Item
from shapely import geometry as shpg

from vibe_core.data import Sentinel1Product, Sentinel1Raster
from vibe_dev.testing.op_tester import OpTester
from vibe_lib.planetary_computer import Sentinel1RTCCollection

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "download_sentinel1.yaml")

IMG_SIZE = 100


@pytest.fixture
def fake_item(tmp_path: Path):
    assets = {}
    for i, band in enumerate(("vh", "vv"), start=1):
        band_path = str(tmp_path / f"{band}.tif")
        with rasterio.open(
            band_path,
            "w",
            driver="GTiff",
            count=1,
            width=IMG_SIZE,
            height=IMG_SIZE,
            dtype="float32",
            nodata=0,
        ) as dst:
            dst.write(i * np.ones((1, IMG_SIZE, IMG_SIZE)))
        assets[band] = Asset(href=band_path)

    return Item(
        id="1",
        geometry=shpg.mapping(shpg.box(0, 0, 1, 1)),
        bbox=None,
        datetime=datetime.now(),
        properties={},
        assets=assets,
    )


@patch.object(pc, "sign")
@patch.object(Sentinel1RTCCollection, "download_item")
@patch.object(Sentinel1RTCCollection, "query_by_id")
@patch("vibe_lib.planetary_computer.get_available_collections", return_value=["sentinel-1-rtc"])
def test_op(
    collections_mock: Mock, query_mock: Mock, download_mock: Mock, sign_mock: Mock, fake_item: Item
):
    query_mock.return_value = fake_item
    download_mock.return_value = [fake_item.assets["vh"].href, fake_item.assets["vv"].href]
    sign_mock.side_effect = lambda x: x
    geom = shpg.box(0, 0, 1, 1)
    fake_input = Sentinel1Product(
        id="1",
        time_range=(datetime.now(), datetime.now()),
        geometry=shpg.mapping(geom),
        assets=[],
        product_name="product_name",
        orbit_number=0,
        relative_orbit_number=0,
        orbit_direction="",
        platform="",
        extra_info={},
        sensor_mode="",
        polarisation_mode="",
    )

    op_tester = OpTester(CONFIG_PATH)
    op_tester.update_parameters({"num_workers": 1})
    out = op_tester.run(sentinel_product=fake_input)
    key = "downloaded_product"
    assert key in out
    product = out[key]
    assert isinstance(product, Sentinel1Raster)
    assert product.time_range == fake_input.time_range
    assert product.geometry == fake_input.geometry
    with rasterio.open(product.raster_asset.local_path) as src:
        profile = src.profile
        ar = src.read()
    assert profile["dtype"] == "float32"
    assert profile["nodata"] == 0.0
    assert ar.shape == (2, IMG_SIZE, IMG_SIZE)
