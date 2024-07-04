import os
from datetime import datetime, timezone
from typing import List, cast

from shapely.geometry import Polygon, box, mapping

from vibe_core.data import DataVibe, DemProduct
from vibe_core.data.core_types import BaseVibe
from vibe_dev.testing.op_tester import OpTester

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "list_dem_products.yaml")


def test_op():
    latitude = 44.0005556
    longitude = -97.0005556
    buffer = 0.1
    bbox = [
        longitude - buffer,
        latitude - buffer,
        longitude + buffer,
        latitude + buffer,
    ]
    polygon: Polygon = box(*bbox, ccw=True)
    start_date = datetime(year=2018, month=2, day=1, tzinfo=timezone.utc)
    end_date = datetime(year=2021, month=2, day=11, tzinfo=timezone.utc)
    input_items = [DataVibe("input_item", (start_date, end_date), mapping(polygon), [])]

    output_data = OpTester(CONFIG_PATH).run(input_items=cast(List[BaseVibe], input_items))

    # Get op result
    output_name = "dem_products"
    assert output_name in output_data
    output_product = output_data[output_name]
    assert isinstance(output_product, list)
    assert len(cast(List[DemProduct], output_data["dem_products"])) == 4
