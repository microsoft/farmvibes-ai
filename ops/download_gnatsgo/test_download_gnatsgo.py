import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from pystac import Asset, Item
from shapely.geometry import Point, mapping

from vibe_core.data import GNATSGOProduct
from vibe_core.data.rasters import GNATSGORaster
from vibe_dev.testing.op_tester import OpTester
from vibe_lib.planetary_computer import GNATSGOCollection

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_gnatsgo.yaml")
INVALID_VARIABLE = "🙅"
FAKE_DATE = datetime(year=2020, month=7, day=1, tzinfo=timezone.utc)


def fake_item():
    assets = {f"{var}": Asset(href=f"fake_href_{var}") for var in GNATSGOCollection.asset_keys}
    return Item(
        id="fake_id",  # type: ignore
        geometry=None,
        bbox=None,
        datetime=None,
        properties={
            "start_datetime": FAKE_DATE.isoformat() + "Z",
            "end_datetime": FAKE_DATE.isoformat() + "Z",
        },
        assets=assets,
    )


@pytest.mark.parametrize("variable", GNATSGOCollection.asset_keys)
@patch("vibe_lib.raster.compress_raster")
@patch("vibe_lib.planetary_computer.get_available_collections", return_value=["gnatsgo-rasters"])
@patch.object(GNATSGOCollection, "download_asset")
@patch.object(GNATSGOCollection, "query_by_id")
def test_op(query: MagicMock, download: MagicMock, _: MagicMock, __: MagicMock, variable: str):
    queried_item = fake_item()
    query.return_value = queried_item
    download.return_value = "/tmp/test.tif"

    polygon = Point(-92.99900, 42.03580).buffer(0.1, cap_style=3)

    input_product = GNATSGOProduct(
        id="conus_101445_2236065_265285_2072225",
        time_range=(FAKE_DATE, FAKE_DATE),
        geometry=mapping(polygon),  # type: ignore
        assets=[],
    )

    op_tester = OpTester(CONFIG_PATH)
    op_tester.update_parameters({"variable": variable})
    output_data = op_tester.run(**{"gnatsgo_product": input_product})

    # Get op result
    output_name = "downloaded_raster"
    assert output_name in output_data
    output_raster = output_data[output_name]
    assert isinstance(output_raster, GNATSGORaster)
    assert output_raster.variable == variable
    assert len(output_raster.bands) == 1
    assert download.call_args.args[0] == queried_item.assets[variable]


def test_op_fails_invalid_variable():
    op_tester = OpTester(CONFIG_PATH)
    op_tester.update_parameters({"variable": INVALID_VARIABLE})
    with pytest.raises(ValueError):
        op_tester.run(input_item=[])
