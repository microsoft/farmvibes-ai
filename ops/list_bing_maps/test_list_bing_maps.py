import os
from datetime import datetime
from typing import List, Optional, cast
from unittest.mock import MagicMock, patch

import pytest
from shapely.geometry import Polygon, box, mapping

from vibe_core.data import DataVibe
from vibe_core.data.products import BingMapsProduct
from vibe_dev.testing.op_tester import OpTester
from vibe_lib.bing_maps import BingMapsCollection

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "list_bing_maps.yaml")
FAKE_TIME_RANGE = (datetime.now(), datetime.now())

# Geometries
WORLD_GEOMETRY = box(-180, -90, 180, 90)
WESTERN_HEMISPHERE_GEOMETRY = box(-180, -90, -0.00001, 90)
EASTERN_HEMISPHERE_GEOMETRY = box(0.00001, -90, 180, 90)
NORTHERN_HEMISPHERE_GEOMETRY = box(-180, 0.00001, 180, 90)
SOUTHERN_HEMISPHERE_GEOMETRY = box(-180, -90, 180, -0.00001)
QUARTER_WORLD_CENTERED_GEOMETRY = box(-89.99999, -44.99999, 89.99999, 44.99999)

FIELD_GEOMETRY = Polygon(
    [
        (-118.940490, 46.998848),
        (-118.876148, 46.998848),
        (-118.876148, 47.013422),
        (-118.940490, 47.013422),
    ]
)


@pytest.mark.parametrize(
    "input_geometry, zoom_level, num_tiles",
    [  # Whole world geometry
        (WORLD_GEOMETRY, zoom_level, n_tiles)
        for zoom_level, n_tiles in [(1, 4), (2, 16), (3, 64), (5, 1024), (7, 16384)]
    ]
    + [  # Half world geometries
        (geom, zoom_level, n_tiles)
        for geom in [
            WESTERN_HEMISPHERE_GEOMETRY,
            EASTERN_HEMISPHERE_GEOMETRY,
            NORTHERN_HEMISPHERE_GEOMETRY,
            SOUTHERN_HEMISPHERE_GEOMETRY,
        ]
        for zoom_level, n_tiles in [(1, 2), (2, 8), (3, 32), (5, 512), (7, 8192)]
    ]
    + [  # Quarter world geometry
        (QUARTER_WORLD_CENTERED_GEOMETRY, zoom_level, n_tiles)
        for zoom_level, n_tiles in [(1, 4), (2, 4), (3, 16), (5, 160), (7, 2304)]
    ]
    + [  # Small field geometry
        (FIELD_GEOMETRY, zoom_level, n_tiles)
        for zoom_level, n_tiles in [
            (1, 1),
            (10, 1),
            (12, 2),
            (14, 8),
            (15, 21),
            (18, 816),
        ]
    ],
)
@patch.object(
    BingMapsCollection,
    "get_download_url_and_subdomains",
    return_value=(
        "fake_download_url_{subdomain}_{quadkey}_{api_key}",
        ["fake_subdomain"],
    ),
)
@patch("vibe_lib.bing_maps.tile_is_available", return_value=True)
def test_list_bing_maps(
    _: MagicMock,
    __: MagicMock,
    input_geometry: Polygon,
    zoom_level: int,
    num_tiles: int,
):
    user_input = DataVibe("user_input", FAKE_TIME_RANGE, mapping(input_geometry), [])

    op_tester = OpTester(CONFIG_PATH)
    op_tester.update_parameters(
        {
            "api_key": "valid_fake_api_key",
            "zoom_level": zoom_level,
            "imagery_set": "Aerial",
            "map_layer": "Basemap",
            "orientation": None,
        }
    )
    output_data = op_tester.run(user_input=user_input)

    # Get op result
    output_name = "products"
    assert output_name in output_data
    output_product = output_data[output_name]
    assert isinstance(output_product, list)
    assert len(cast(List[BingMapsProduct], output_data["products"])) == num_tiles


@pytest.mark.parametrize(
    "zoom_level, api_key, imagery_set, map_layer, orientation",
    [
        # Invalid api_key
        (10, "", "Aerial", "Basemap", None),
        (10, None, "Aerial", "Basemap", None),
        # Invalid zoom_level
        (0, "valid_fake_api_key", "Aerial", "Basemap", None),
        (21, "valid_fake_api_key", "Aerial", "Basemap", None),
        # Invalid imagery_set
        (10, "valid_fake_api_key", "invalid_imagery_set", "Basemap", None),
        # Invalid map_layer
        (10, "valid_fake_api_key", "Aerial", "invalid_map_layer", None),
        # Invalid orientation
        (10, "valid_fake_api_key", "Aerial", "Basemap", -1),
        (10, "valid_fake_api_key", "Aerial", "Basemap", 180),
        (10, "valid_fake_api_key", "Aerial", "Basemap", 380),
    ],
)
def test_invalid_parameters(
    zoom_level: int,
    api_key: str,
    imagery_set: str,
    map_layer: str,
    orientation: Optional[float],
):
    user_input = DataVibe("user_input", FAKE_TIME_RANGE, mapping(FIELD_GEOMETRY), [])

    op_tester = OpTester(CONFIG_PATH)

    op_tester.update_parameters(
        {
            "api_key": api_key,
            "zoom_level": zoom_level,
            "imagery_set": imagery_set,
            "map_layer": map_layer,
            "orientation": orientation,
        }
    )
    with pytest.raises(ValueError):
        op_tester.run(user_input=user_input)
