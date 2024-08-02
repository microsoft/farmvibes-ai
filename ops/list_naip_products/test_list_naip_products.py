# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from datetime import datetime, timezone
from typing import List, cast

from shapely.geometry import Polygon, box, mapping

from vibe_core.data import DataVibe, DemProduct
from vibe_dev.testing.op_tester import OpTester

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "list_naip_products.yaml")


def test_op():
    latitude = 42.21422
    longitude = -93.22890
    buffer = 0.001
    bbox = [longitude - buffer, latitude - buffer, longitude + buffer, latitude + buffer]
    polygon: Polygon = box(*bbox, ccw=True)
    start_date = datetime(year=2018, month=2, day=1, tzinfo=timezone.utc)
    end_date = datetime(year=2021, month=2, day=11, tzinfo=timezone.utc)
    input_item = DataVibe("input_item", (start_date, end_date), mapping(polygon), [])

    output_data = OpTester(CONFIG_PATH).run(input_item=input_item)

    # Get op result
    output_name = "naip_products"
    assert output_name in output_data
    output_product = output_data[output_name]
    assert isinstance(output_product, list)
    assert len(cast(List[DemProduct], output_data["naip_products"])) == 1
