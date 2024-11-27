# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import inspect
import typing
from datetime import datetime
from unittest.mock import MagicMock, patch

import orjson
import pytest

import vibe_core.data
from vibe_core.data.utils import StacConverter, deserialize_stac, serialize_stac

BASIC_MOCK_VALUES = {
    int: 42,
    float: 42.0,
    str: "mock_str",
    bool: True,
    datetime: datetime.now(),
}

DATAVIBES_MOCK_FIELDS = {
    "id": "mock_id",
    "time_range": (datetime.now(), datetime.now()),
    "geometry": {"type": "Point", "coordinates": [0, 0]},
    "assets": [],
}

FARMVIBES_DATA_CLASSES = [
    getattr(vibe_core.data, name)
    for name in dir(vibe_core.data)
    if inspect.isclass(getattr(vibe_core.data, name))
    and issubclass(getattr(vibe_core.data, name), vibe_core.data.DataVibe)
]


def is_optional(t: type) -> bool:
    return typing.get_origin(t) is typing.Union and type(None) in typing.get_args(t)  # type: ignore


def create_mock_instance(cls: type) -> typing.Any:
    if cls in BASIC_MOCK_VALUES:
        return BASIC_MOCK_VALUES[cls]  # type: ignore

    args = {}
    params = {
        **inspect.signature(cls.__init__).parameters,
        **inspect.signature(cls.__new__).parameters,
    }
    for name, param in params.items():
        if name in ["self", "args", "kwargs", "_cls"]:
            continue
        elif name in DATAVIBES_MOCK_FIELDS:
            args[name] = DATAVIBES_MOCK_FIELDS[name]
        else:
            args[name] = create_mock_value(param.annotation)
    return cls(**args)


def create_mock_value(tp: type) -> typing.Any:
    # Handle basic types with random or default values
    if tp in BASIC_MOCK_VALUES:
        return BASIC_MOCK_VALUES[tp]  # type: ignore
    elif tp is list or getattr(tp, "__origin__", None) is list:
        return []
    elif tp is tuple or getattr(tp, "__origin__", None) is tuple:
        # Create an empty tuple or a tuple with mock values if types are specified
        return tuple(create_mock_value(arg) for arg in getattr(tp, "__args__", []))
    elif tp is dict or getattr(tp, "__origin__", None) is dict:
        return {}
    elif tp is typing.Any:
        return None
    elif is_optional(tp):
        # check which type is optional and create a mock value for it
        return create_mock_value(tp.__args__[0])  # type: ignore
    elif inspect.isclass(tp):
        # Recursively create instances for complex types
        return create_mock_instance(tp)

    raise NotImplementedError(f"Mocking not implemented for type: {tp}")


@patch.object(vibe_core.data.HansenProduct, "validate_url", return_value=True)
@pytest.mark.parametrize("cls", FARMVIBES_DATA_CLASSES)
def test_serialization_deserialization(
    _: MagicMock,
    cls: type,
):
    converter = StacConverter()

    mock_instance = create_mock_instance(cls)
    stac_item = converter.to_stac_item(mock_instance)

    json_instance = orjson.loads(orjson.dumps(serialize_stac(stac_item)))
    deserialized_stac_item = deserialize_stac(json_instance)
    deserialized = converter.from_stac_item(deserialized_stac_item)
    assert mock_instance == deserialized

    deserialized = converter.from_stac_item(stac_item)
    assert mock_instance == deserialized
