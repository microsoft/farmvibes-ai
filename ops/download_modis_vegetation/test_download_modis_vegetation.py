import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from pystac import Asset, Item
from shapely import geometry as shpg

from vibe_core.data import ModisProduct, Raster
from vibe_dev.testing.op_tester import OpTester
from vibe_lib.planetary_computer import Modis16DayVICollection

HERE = os.path.dirname(os.path.abspath(__file__))
INDICES = ("ndvi", "evi")
FAKE_TIME_RANGE = (datetime(2020, 11, 1), datetime(2020, 11, 2))
INVALID_INDEX = "🙅"


def fake_items(resolution: int):
    assets = {
        f"250m_16_days_{index.upper()}": Asset(href=f"fake_href_{resolution}_{index}")
        for index in INDICES
    }
    return [
        Item(
            id=f"{resolution}m-id",  # type: ignore
            geometry=None,
            bbox=None,
            datetime=None,
            properties={
                "start_datetime": FAKE_TIME_RANGE[0].isoformat() + "Z",
                "end_datetime": FAKE_TIME_RANGE[1].isoformat() + "Z",
            },
            assets=assets,
        )
    ]


@pytest.mark.parametrize("resolution", (250, 500))
@pytest.mark.parametrize("index", ("ndvi", "evi"))
@patch("vibe_lib.planetary_computer.get_available_collections")
@patch.object(Modis16DayVICollection, "download_asset")
@patch.object(Modis16DayVICollection, "query")
def test_op(
    query: MagicMock,
    download_asset: MagicMock,
    get_collections: MagicMock,
    index: str,
    resolution: int,
):
    get_collections.return_value = list(Modis16DayVICollection.collections.values())
    items = fake_items(resolution)
    query.return_value = items
    download_asset.side_effect = lambda asset, path: asset.href

    geom = shpg.Point(1, 1).buffer(0.01, cap_style=3)
    time_range = (datetime(2022, 11, 1), datetime(2022, 11, 2))
    x = ModisProduct(
        id="1", time_range=time_range, geometry=shpg.mapping(geom), resolution=resolution, assets=[]
    )

    op_tester = OpTester(os.path.join(HERE, "download_modis_vegetation.yaml"))
    op_tester.update_parameters({"index": index})
    o = op_tester.run(product=x)

    query.assert_called_once_with(roi=x.bbox, time_range=x.time_range, ids=[x.id])
    download_asset.assert_called_once()
    assert isinstance(o["index"], Raster)
    assert o["index"].raster_asset.local_path == f"fake_href_{resolution}_{index}"


def test_op_fails_invalid_index():
    op_tester = OpTester(os.path.join(HERE, "download_modis_vegetation.yaml"))
    op_tester.update_parameters({"index": INVALID_INDEX})
    with pytest.raises(ValueError):
        op_tester.run(product=None)  # type: ignore
