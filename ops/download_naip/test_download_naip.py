# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from shapely.geometry import Polygon, box, mapping

from vibe_core.data import NaipProduct
from vibe_core.data.rasters import NaipRaster
from vibe_dev.testing.op_tester import OpTester
from vibe_lib.planetary_computer import NaipCollection

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_naip.yaml")


@patch(
    "vibe_lib.planetary_computer.get_available_collections",
    return_value=[NaipCollection.collection],
)
@patch.object(NaipCollection, "query_by_id")
@patch.object(NaipCollection, "download_item", return_value=["/tmp/test.tif"])
def test_op(_: MagicMock, __: MagicMock, ___: MagicMock):
    latitude = 42.21422
    longitude = -93.22890
    buffer = 0.001
    bbox = [longitude - buffer, latitude - buffer, longitude + buffer, latitude + buffer]
    polygon: Polygon = box(*bbox, ccw=True)
    start_date = datetime(year=2018, month=2, day=1, tzinfo=timezone.utc)
    end_date = datetime(year=2021, month=2, day=11, tzinfo=timezone.utc)

    output: NaipProduct = NaipProduct(
        id=str("ia_m_4209355_nw_15_060_20190730_20191105"),
        time_range=(
            start_date,
            end_date,
        ),
        geometry=mapping(polygon),  # type: ignore
        assets=[],
        tile_id=str("ia_m_4209355_nw_15_060_20190730_20191105"),
        resolution=0.6,
        year=2019,
    )

    output_data = OpTester(CONFIG_PATH).run(**{"input_product": output})

    # Get op result
    output_name = "downloaded_product"
    assert output_name in output_data
    output_product = output_data[output_name]
    assert isinstance(output_product, NaipRaster)
