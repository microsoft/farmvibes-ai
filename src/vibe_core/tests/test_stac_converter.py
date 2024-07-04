# pyright: reportUnknownMemberType=false

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
from shapely import geometry as shpg
from shapely.geometry.base import BaseGeometry

from vibe_core.data import AssetVibe, DataVibe, Raster, StacConverter


@dataclass
class DateVibe(DataVibe):
    date_field: datetime
    int_field: int
    str_field: str
    other_field: Any
    other_list: List[str]
    date_list: List[datetime]
    date_dict: Dict[str, datetime]
    date_tuple: Tuple[datetime, datetime, datetime]
    mixed_tuple: Tuple[int, datetime]
    var_tuple: Tuple[datetime, ...]
    nested_list: List[List[datetime]]
    dict_list: Dict[str, List[datetime]]
    super_nest: Dict[Any, List[Dict[Any, Dict[Any, Tuple[datetime, ...]]]]]
    super_nest_no: Dict[Any, List[Dict[Any, Dict[Any, Tuple[Any, ...]]]]]


@dataclass
class ShapeVibe(DataVibe):
    shape: BaseGeometry
    shape_dict: Dict[str, BaseGeometry]


@pytest.fixture
def converter() -> StacConverter:
    return StacConverter()


def test_conversion_roundtrip(converter: StacConverter, tmp_path: Path):
    asset_path = tmp_path.as_posix()
    now = datetime.now()
    geom: Dict[str, Any] = shpg.mapping(shpg.box(-1, -1, 1, 1))
    terravibes_data = DataVibe(id="assetless", time_range=(now, now), geometry=geom, assets=[])
    # Assetless DataVibe conversion
    assert converter.from_stac_item(converter.to_stac_item(terravibes_data)) == terravibes_data
    mimefull = AssetVibe(reference=asset_path, type="image/tiff", id="mimefull")
    terravibes_data.assets.append(mimefull)
    # Conversion with asset that has mimetype
    assert converter.from_stac_item(converter.to_stac_item(terravibes_data)) == terravibes_data
    mimeless = AssetVibe(reference=asset_path, type=None, id="mimeless")
    # Conversion with asset that has no mimetype
    terravibes_data.assets.append(mimeless)
    assert converter.from_stac_item(converter.to_stac_item(terravibes_data)) == terravibes_data


def test_conversion_raster(converter: StacConverter, tmp_path: Path):
    asset_path = tmp_path.as_posix()
    now = datetime.now()
    geom: Dict[str, Any] = shpg.mapping(shpg.box(-1, -1, 1, 1))
    tiff_asset = AssetVibe(reference=asset_path, type="image/tiff", id="tiff_asset")
    json_asset = AssetVibe(reference=asset_path, type="application/json", id="json_asset")
    raster = Raster(
        id="extra_info_test",
        time_range=(now, now),
        geometry=geom,
        assets=[tiff_asset, json_asset],
        bands={"B1": 0, "B2": 1, "B3": 2},
    )
    converted = converter.from_stac_item(converter.to_stac_item(raster))
    assert isinstance(converted, Raster)
    assert converted == raster
    assert raster.raster_asset == converted.raster_asset
    assert raster.visualization_asset == converted.visualization_asset


def test_datetime_field_serialization(converter: StacConverter):
    now = datetime.now()
    geom: Dict[str, Any] = shpg.mapping(shpg.box(-1, -1, 1, 1))
    test_vibe = DateVibe(
        "assetless",
        (now, now),
        geom,
        [],
        now,
        1,
        "1",
        None,
        ["1", "2"],
        [datetime.now() for _ in range(2)],
        {f"{i}": datetime.now() for i in range(3)},
        (datetime.now(), datetime.now(), datetime.now()),
        (1, datetime.now()),
        tuple(datetime.now() for _ in range(4)),
        [[datetime.now()]],
        {"1": [datetime.now() for _ in range(2)], "2": [datetime.now() for _ in range(3)]},
        {0: [{0: {0: (datetime.now(),)}}]},
        {0: [{0: {0: ("NO",)}}]},
    )
    forward = converter.to_stac_item(test_vibe)
    assert forward.properties["date_field"] == now.isoformat()
    round_trip = converter.from_stac_item(forward)
    assert test_vibe == round_trip


def test_geom_field_serialization(converter: StacConverter):
    now = datetime.now()
    geom: Dict[str, Any] = shpg.mapping(shpg.box(-1, -1, 1, 1))
    test_vibe = ShapeVibe(
        "assetless",
        (now, now),
        geom,
        [],
        shpg.box(0, 0, 2, 2),
        {f"{i}": shpg.box(0, 0, i, i) for i in range(1, 5)},
    )
    forward = converter.to_stac_item(test_vibe)
    assert forward.properties["shape"] == {
        "type": "Polygon",
        "coordinates": (((2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0), (2.0, 0.0)),),
    }
    round_trip = converter.from_stac_item(forward)
    assert test_vibe == round_trip
