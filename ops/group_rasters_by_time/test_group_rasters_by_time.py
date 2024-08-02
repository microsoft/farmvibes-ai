# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import mimetypes
import os
from datetime import datetime, timedelta
from typing import List, cast

import pytest
from shapely.geometry import Polygon, box, mapping

from vibe_core.data import Raster, RasterSequence
from vibe_core.data.core_types import AssetVibe, BaseVibe, gen_guid
from vibe_dev.testing.op_tester import OpTester

START_DATE = datetime(2022, 1, 1)
NDAYS = 730  # 2 years
EXPECTED = [("day_of_year", 365), ("week", 52), ("month", 12), ("year", 2), ("month_and_year", 24)]

YAML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "group_rasters_by_time.yaml")


@pytest.mark.parametrize("criterion, expected", EXPECTED)
def test_op(criterion: str, expected: int):
    op_tester = OpTester(YAML_PATH)
    op_tester.update_parameters({"criterion": criterion})

    latitude = 42.0
    longitude = 42.0
    buffer = 0.0042
    bbox = [longitude - buffer, latitude - buffer, longitude + buffer, latitude + buffer]
    polygon: Polygon = box(*bbox, ccw=True)

    fake_asset = AssetVibe(reference="", type=mimetypes.types_map[".tif"], id="fake_asset")

    rasters = [
        Raster(
            id=gen_guid(),
            time_range=(START_DATE + timedelta(i), START_DATE + timedelta(i)),
            geometry=mapping(polygon),
            assets=[fake_asset],
            bands={},
        )
        for i in range(NDAYS)
    ]

    res = cast(
        List[RasterSequence], op_tester.run(rasters=cast(List[BaseVibe], rasters))["raster_groups"]
    )
    assert len(res) == expected
