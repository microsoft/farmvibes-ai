import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from shapely.geometry import Point, mapping

from vibe_core.data import ClimatologyLabProduct
from vibe_dev.testing.op_tester import OpTester

FAKE_GEOMETRY = Point(-92.99900, 42.03580).buffer(0.1, cap_style=3)
FAKE_TIME_RANGE = (
    datetime(year=2019, month=1, day=1, tzinfo=timezone.utc),
    datetime(year=2019, month=12, day=31, tzinfo=timezone.utc),
)

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "download_climatology_lab.yaml"
)


@patch("vibe_core.file_downloader.download_file")
def test_op(_: MagicMock):
    input_product = ClimatologyLabProduct(
        id="fake_product",
        time_range=FAKE_TIME_RANGE,
        geometry=mapping(FAKE_GEOMETRY),  # type: ignore
        assets=[],
        url="fake_href",
        variable="fake_variable",
    )

    op_tester = OpTester(CONFIG_PATH)
    output_data = op_tester.run(**{"input_product": input_product})

    # Get op result
    output_name = "downloaded_product"
    assert output_name in output_data
    output_raster = output_data[output_name]
    assert isinstance(output_raster, ClimatologyLabProduct)
    assert len(output_raster.assets) == 1
