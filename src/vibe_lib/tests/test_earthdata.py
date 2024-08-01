# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
from datetime import datetime
from typing import Any, Optional, Tuple, cast
from unittest.mock import Mock, patch

import pytest
import requests
from shapely import geometry as shpg

from vibe_core.data.core_types import BBox
from vibe_lib.earthdata import EarthDataAPI, format_geometry

FMT_BOX = "2.0,1.0,2.0,3.0,0.0,3.0,0.0,1.0,2.0,1.0"
PROCESSING_LEVEL = "GEDI02_B.002"


@pytest.fixture
def test_box():
    return shpg.box(0, 1, 2, 3)


def fake_responses(num_items: int, page_size: int):
    def foo(*args: Any, **kwargs: Any):
        nonlocal num_items
        num_return = min(num_items, page_size)
        num_items = num_items - num_return
        return {"feed": {"entry": [None for _ in range(num_return)]}}

    return foo


def test_format_geometry(test_box: shpg.Polygon):
    fmt_geoms = format_geometry(test_box)
    assert len(fmt_geoms) == 1
    assert fmt_geoms[0] == FMT_BOX


def test_format_cw_geometry(test_box: shpg.Polygon):
    # Make sure we orient geometry properly (counter-clockwise)
    test_geom = shpg.polygon.orient(test_box, sign=-1)
    fmt_cw = format_geometry(test_geom)[0]
    assert fmt_cw == FMT_BOX


def test_format_multipoly(test_box: shpg.Polygon):
    test_geom = cast(shpg.MultiPolygon, test_box.union(shpg.box(10, 10, 11, 11)))
    fmt_geoms = format_geometry(test_geom)
    assert len(fmt_geoms) == 2
    assert fmt_geoms[0] == FMT_BOX


def test_api_wrapper_base_payload():
    api = EarthDataAPI(PROCESSING_LEVEL)
    payload = api._get_payload(geometry=None, bbox=None, time_range=None, id=None)
    assert len(payload) == 3
    assert payload["provider"] == api.provider
    assert payload["concept_id"] == api.concept_ids[PROCESSING_LEVEL]
    assert payload["page_size"] == api.page_size


@pytest.mark.parametrize("id", (None, "test_id"))
@pytest.mark.parametrize("time_range", (None, (datetime.now(), datetime.now())))
@pytest.mark.parametrize("bbox", (None, (0, 0, 1, 1)))
@pytest.mark.parametrize("geometry", (None, shpg.box(0, 0, 1, 1)))
def test_api_wrapper_payload_keys(
    geometry: Optional[shpg.Polygon],
    bbox: Optional[BBox],
    time_range: Optional[Tuple[datetime, datetime]],
    id: Optional[str],
):
    api = EarthDataAPI(PROCESSING_LEVEL)
    payload = api._get_payload(geometry=geometry, bbox=bbox, time_range=time_range, id=id)
    if geometry is not None:
        assert "polygon[]" in payload
        assert "options[polygon][or]" in payload
    if bbox is not None:
        assert "bounding_box" in payload
    if time_range is not None:
        assert "temporal" in payload
    if id is not None:
        assert "producer_granule_id" in payload


@pytest.mark.parametrize("num_items", (1, 2000, 2001, 9000))
@patch.object(requests, "post")
def test_api_wrapper_paging(post: Mock, num_items: int):
    api = EarthDataAPI(PROCESSING_LEVEL)
    response_mock = Mock()
    response_mock.configure_mock(**{"json.side_effect": fake_responses(num_items, api.page_size)})
    post.return_value = response_mock
    api.query()
    expected_calls = math.ceil((num_items + 1) / api.page_size)
    assert post.call_count == expected_calls
    for i, call_args in enumerate(post.call_args_list, 1):
        assert call_args[1]["data"]["pageNum"] == i


@patch.object(requests, "post")
def test_api_wrapper_max_pages(post: Mock):
    api = EarthDataAPI(PROCESSING_LEVEL)
    response_mock = Mock()
    response_mock.configure_mock(
        **{"json.side_effect": fake_responses(api.max_items, api.page_size)}
    )
    post.return_value = response_mock
    api.query()

    response_mock.configure_mock(
        **{"json.side_effect": fake_responses(api.max_items + api.page_size, api.page_size)}
    )

    with pytest.raises(RuntimeError):
        api.query()
