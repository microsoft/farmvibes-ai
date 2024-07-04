import mimetypes
import os
from datetime import datetime, timezone
from typing import List, Tuple

import pytest
from shapely.geometry import Polygon, box, mapping, shape

from vibe_core.data import AssetVibe, Raster, RasterSequence, gen_guid
from vibe_dev.testing.op_tester import OpTester

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "list_to_sequence.yaml")

# Geometries
WORLD_GEOM = box(-90, -180, 90, 180)
WESTERN_HEMS_GEOM = box(-90, -180, 90, 0.0)
EASTERN_HEMS_GEOM = box(-90, 0.0, 90, 180)
NORTHERN_HEMS_GEOM = box(0.0, -180, 90, 180)
SOUTHERN_HEMS_GEOM = box(-90, -180, 0.0, 180)
NW_REGION_GEOM = box(0.0, -180, 90, 0.0)
FAKE_GEOMETRY = box(-5.0, -5.0, -1.0, -1.0)  # SW

# Time ranges
FAKE_TIME_RANGE = (datetime.now(tz=timezone.utc), datetime.now(tz=timezone.utc))
TR_1900s = (
    datetime(1900, 1, 1, tzinfo=timezone.utc),
    datetime(1999, 12, 31, tzinfo=timezone.utc),
)
TR_1990s = (
    datetime(1990, 1, 1, tzinfo=timezone.utc),
    datetime(1999, 12, 31, tzinfo=timezone.utc),
)
TR_2000s = (
    datetime(2000, 1, 1, tzinfo=timezone.utc),
    datetime(2009, 12, 31, tzinfo=timezone.utc),
)
TR_1900s_2000s = (
    datetime(1900, 1, 1, tzinfo=timezone.utc),
    datetime(2009, 12, 31, tzinfo=timezone.utc),
)


def create_raster(geometry: Polygon, time_range: Tuple[datetime, datetime]) -> Raster:
    return Raster(
        id=gen_guid(),
        time_range=time_range,
        geometry=mapping(geometry),
        assets=[AssetVibe(reference="", type=mimetypes.types_map[".tif"], id=gen_guid())],
        bands={},
    )


@pytest.mark.parametrize(
    "input_geometry_list, input_time_range_list, expected_geometry",
    [
        ([NORTHERN_HEMS_GEOM, SOUTHERN_HEMS_GEOM], [FAKE_TIME_RANGE] * 2, WORLD_GEOM),
        ([WESTERN_HEMS_GEOM, EASTERN_HEMS_GEOM], [FAKE_TIME_RANGE] * 2, WORLD_GEOM),
        ([WESTERN_HEMS_GEOM, NW_REGION_GEOM], [FAKE_TIME_RANGE] * 2, WESTERN_HEMS_GEOM),
        ([FAKE_GEOMETRY], [FAKE_TIME_RANGE], FAKE_GEOMETRY),
    ],
)
def test_geometry_combination(
    input_geometry_list: List[Polygon],
    input_time_range_list: List[Tuple[datetime, datetime]],
    expected_geometry: Polygon,
):
    rasters = [
        create_raster(geometry, tr)
        for geometry, tr in zip(input_geometry_list, input_time_range_list)
    ]

    op_tester = OpTester(CONFIG_PATH)
    output_data = op_tester.run(list_rasters=rasters)  # type: ignore

    # Get op result
    output_name = "rasters_seq"
    assert output_name in output_data
    output_seq = output_data[output_name]
    assert type(output_seq) is RasterSequence
    assert len(output_seq.asset_geometry) == len(rasters)
    assert expected_geometry.equals(shape(output_seq.geometry))


@pytest.mark.parametrize(
    "input_time_range_list, expected_time_range",
    [
        ([TR_1900s, TR_2000s], TR_1900s_2000s),
        ([TR_1900s, TR_1990s], TR_1900s),
        ([FAKE_TIME_RANGE], FAKE_TIME_RANGE),
    ],
)
def test_time_range_combination(
    input_time_range_list: List[Tuple[datetime, datetime]],
    expected_time_range: Tuple[datetime, datetime],
):
    rasters = [create_raster(FAKE_GEOMETRY, time_range) for time_range in input_time_range_list]

    op_tester = OpTester(CONFIG_PATH)
    output_data = op_tester.run(list_rasters=rasters)  # type: ignore

    # Get op result
    output_name = "rasters_seq"
    assert output_name in output_data
    output_seq = output_data[output_name]
    assert type(output_seq) is RasterSequence
    assert len(output_seq.asset_time_range) == len(rasters)
    assert output_seq.time_range == expected_time_range
