# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from datetime import datetime
from typing import Any

import pytest

from vibe_common.vibe_dapr_client import VibeDaprClient
from vibe_core.datamodel import Message, SpatioTemporalJson


class MockResponse:
    def __init__(self, content: Any):
        self._content = content

    async def json(self, loads: Any, **kwargs: Any) -> Any:
        return loads(self._content, **kwargs)


def test_state_store_dumps_dataclass():
    client = VibeDaprClient()
    assert client._dumps(Message(message="hi", id=None, location=None))


def test_state_store_fails_to_dump_pydantic_model_with_invalid_values():
    client = VibeDaprClient()
    with pytest.raises(ValueError):
        client._dumps(
            SpatioTemporalJson(
                start_date=datetime.now(),
                end_date=datetime.now(),
                geojson={"location": float("nan")},
            )
        )


def test_state_store_float_serialized_as_str():
    lat = -52.6324171000924
    lon = -7.241144827812494
    test_input = SpatioTemporalJson(
        start_date=datetime.now(),
        end_date=datetime.now(),
        geojson={"coordinates": [lat, lon]},
    )
    client = VibeDaprClient()
    test_input_json = client.obj_json(test_input)
    assert test_input_json["geojson"]["coordinates"][0] == repr(lat)
    assert test_input_json["geojson"]["coordinates"][1] == repr(lon)


@pytest.mark.anyio
async def test_state_store_response_deserialize_floats():
    lat = -52.6324171000924
    lon = -7.241144827812494

    test_response = MockResponse(
        str.encode('{{"geojson": {{"coordinates": ["{0}", "{1}"]}}}}'.format(lat, lon))
    )

    client = VibeDaprClient()
    test_response_json = await client.response_json(test_response)  # type: ignore
    assert isinstance(test_response_json["geojson"]["coordinates"][0], float)
    assert isinstance(test_response_json["geojson"]["coordinates"][1], float)
    assert test_response_json["geojson"]["coordinates"][0] == lat
    assert test_response_json["geojson"]["coordinates"][1] == lon
