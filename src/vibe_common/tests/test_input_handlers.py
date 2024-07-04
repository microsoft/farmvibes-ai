from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest

from vibe_common.input_handlers import gen_stac_item_from_bounds, handle_non_collection


def test_with_feature_geojson():
    start_date = datetime.now(timezone.utc)
    end_date = start_date - timedelta(days=6 * 30)

    test_feature: Dict[str, Any] = {
        "type": "Feature",
        "properties": {"Name": "some_name"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-118.675944, 46.916908],
                    [-118.675944, 46.79631],
                    [-118.841574, 46.79631],
                    [-118.841574, 46.916908],
                    [-118.675944, 46.916908],
                ]
            ],
        },
    }

    item = handle_non_collection(test_feature, start_date, end_date)

    assert item["properties"]["start_datetime"] == start_date.isoformat()
    assert item["properties"]["end_datetime"] == end_date.isoformat()
    assert item["geometry"] == test_feature["geometry"]


def test_with_geometry_geojson():
    start_date = datetime.now(timezone.utc)
    end_date = start_date - timedelta(days=6 * 30)

    test_geometry: Dict[str, Any] = {
        "type": "Polygon",
        "name": "some_name",
        "coordinates": [
            [
                [-85.34557342529297, 37.441882193395124],
                [-85.18661499023436, 37.441882193395124],
                [-85.18661499023436, 37.53804390907164],
                [-85.34557342529297, 37.53804390907164],
                [-85.34557342529297, 37.441882193395124],
            ]
        ],
    }

    item = handle_non_collection(test_geometry, start_date, end_date)

    assert item["properties"]["start_datetime"] == start_date.isoformat()
    assert item["properties"]["end_datetime"] == end_date.isoformat()
    assert item["geometry"] == test_geometry


@patch("vibe_common.input_handlers.handle_non_collection")
def test_with_feature_collection_geojson(mock_handle: Mock):
    start_date = datetime.now(timezone.utc)
    end_date = start_date - timedelta(days=6 * 30)

    test_feature = {
        "type": "Feature",
        "properties": {"Name": "some_name"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-118.675944, 46.916908],
                    [-118.675944, 46.79631],
                    [-118.841574, 46.79631],
                    [-118.841574, 46.916908],
                    [-118.675944, 46.916908],
                ]
            ],
        },
    }

    test_collection: Dict[str, Any] = {
        "type": "FeatureCollection",
        "name": "some_name",
        "features": [test_feature],
    }

    gen_stac_item_from_bounds(test_collection, start_date, end_date)

    mock_handle.assert_called_once_with(test_feature, start_date, end_date)

    test_collection["features"].append(test_feature)
    with pytest.raises(ValueError):
        gen_stac_item_from_bounds(test_collection, start_date, end_date)
