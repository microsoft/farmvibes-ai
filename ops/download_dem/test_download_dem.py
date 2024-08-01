# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from shapely.geometry import Polygon, box, mapping

from vibe_core.data import DemProduct
from vibe_core.data.rasters import DemRaster
from vibe_dev.testing.op_tester import OpTester
from vibe_lib.planetary_computer import USGS3DEPCollection

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_dem.yaml")


@patch(
    "vibe_lib.planetary_computer.get_available_collections",
    return_value=[USGS3DEPCollection.collection],
)
@patch.object(USGS3DEPCollection, "query_by_id")
@patch(
    "vibe_lib.planetary_computer.USGS3DEPCollection.download_item", return_value=["/tmp/test.tif"]
)
def test_op(_: MagicMock, __: MagicMock, ___: MagicMock):
    latitude = 44.0005556
    longitude = -97.0005556
    buffer = 0.1
    bbox = [longitude - buffer, latitude - buffer, longitude + buffer, latitude + buffer]
    polygon: Polygon = box(*bbox, ccw=True)
    start_date = datetime(year=2021, month=2, day=1, tzinfo=timezone.utc)
    end_date = datetime(year=2021, month=2, day=11, tzinfo=timezone.utc)

    output = DemProduct(
        id=str("n44w098-13"),
        time_range=(
            start_date,
            end_date,
        ),
        geometry=mapping(polygon),
        assets=[],
        tile_id=str("n44w098-13"),
        resolution=10,
        provider=str("USGS3DEP"),
    )

    output_data = OpTester(CONFIG_PATH).run(input_product=output)

    # Get op result
    output_name = "downloaded_product"
    assert output_name in output_data
    output_product = output_data[output_name]
    assert isinstance(output_product, DemRaster)
