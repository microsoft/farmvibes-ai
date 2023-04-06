from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple, cast

import pytest
from shapely import geometry as shpg

from vibe_core.data import CategoricalRaster, Raster, RasterSequence
from vibe_core.data.core_types import BaseVibe, DataVibe, DataVibeType, TimeSeries
from vibe_core.data.utils import (
    StacConverter,
    get_most_specific_type,
    serialize_input,
    serialize_stac,
)


@dataclass
class BaseTest(BaseVibe):
    int_field: int
    str_field: str


def serialize_vibe(vibe: BaseVibe):
    converter = StacConverter()
    return serialize_stac(converter.to_stac_item(vibe))


@pytest.fixture
def input_pair(request: pytest.FixtureRequest):
    param: str = request.param  # type:ignore
    if param == "base":
        x = BaseTest(0, "string")
        return x, serialize_vibe(x)
    now = datetime.now()
    geom = shpg.mapping(shpg.box(0, 0, 1, 1))
    kwargs = {"id": "1", "time_range": (now, now), "geometry": geom, "assets": []}
    if param == "vibe":
        x = DataVibe(**kwargs)
    elif param == "raster":
        x = Raster(**kwargs, bands={})
    else:
        raise ValueError(f"Unrecognized parameter {param}")
    return x, serialize_vibe(x)


@pytest.mark.parametrize("repeats", (1, 10))
@pytest.mark.parametrize("input_pair", ("base", "vibe", "raster"), indirect=True)
def test_serialize_basevibe(input_pair: Tuple[BaseVibe, Dict[str, Any]], repeats: int):
    input, serial = input_pair
    assert serialize_input(input) == serial
    input_list = [input for _ in range(repeats)]
    serial_list = [serial for _ in range(repeats)]
    assert serialize_input(input_list) == serial_list
    assert serialize_input({"item": input, "list": input_list}) == {
        "item": serial,
        "list": serial_list,
    }


def test_get_most_specific_type():
    assert get_most_specific_type([DataVibe, DataVibe]) is DataVibe
    assert get_most_specific_type([DataVibe, TimeSeries]) is TimeSeries
    assert get_most_specific_type([DataVibe, Raster]) is Raster
    assert get_most_specific_type([CategoricalRaster, Raster]) is CategoricalRaster
    assert get_most_specific_type([DataVibe, Raster, CategoricalRaster]) is CategoricalRaster


def test_most_specific_type_incompatible_fails():
    with pytest.raises(ValueError):
        get_most_specific_type([DataVibe, Raster, TimeSeries])
    with pytest.raises(ValueError):
        get_most_specific_type([Raster, TimeSeries])
    with pytest.raises(ValueError):
        get_most_specific_type([RasterSequence, CategoricalRaster])


def test_most_specific_type_container():
    assert get_most_specific_type(cast(List[DataVibeType], [DataVibe, List[DataVibe]])) is DataVibe
    assert (
        get_most_specific_type(cast(List[DataVibeType], [List[DataVibe], List[DataVibe]]))
        is List[DataVibe]
    )
    assert (
        get_most_specific_type(cast(List[DataVibeType], [List[CategoricalRaster], Raster]))
        is CategoricalRaster
    )
    assert (
        get_most_specific_type(cast(List[DataVibeType], [List[CategoricalRaster], List[Raster]]))
        is List[CategoricalRaster]
    )


@pytest.mark.parametrize("input_pair", ("base", "vibe", "raster"), indirect=True)
def test_serialize_deserialize_stac(input_pair: Tuple[BaseVibe, Dict[str, Any]]):
    input, serial = input_pair
    converter = StacConverter()
    serialized = converter.to_stac_item(input)
    output = converter.from_stac_item(serialized)

    assert input == output
    assert serial == serialize_stac(serialized)
