import os
from datetime import datetime, timezone
from typing import List, cast
from unittest.mock import MagicMock, patch

import pytest
from shapely.geometry import Point, mapping

from vibe_core.data import DataVibe
from vibe_core.data.products import ClimatologyLabProduct
from vibe_dev.testing.op_tester import OpTester
from vibe_lib.climatology_lab import (
    ClimatologyLabCollection,
    GridMETCollection,
    TerraClimateCollection,
)

TERRACLIMATE_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "list_terraclimate.yaml"
)
GRIDMET_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "list_gridmet.yaml")

FAKE_GEOMETRY = Point(-92.99900, 42.03580).buffer(0.1, cap_style=3)
FAKE_TIME_RANGE = (
    datetime(year=2019, month=1, day=1, tzinfo=timezone.utc),
    datetime(year=2020, month=12, day=31, tzinfo=timezone.utc),
)

INVALID_VARIABLE = "🙅"


@pytest.mark.parametrize(
    "config_path, variable",
    [
        (p, v)
        for p, c in [
            (TERRACLIMATE_CONFIG_PATH, TerraClimateCollection),
            (GRIDMET_CONFIG_PATH, GridMETCollection),
        ]
        for v in c.asset_keys
    ],
)
@patch.object(ClimatologyLabCollection, "check_url_variable_year", return_value=True)
def test_gridmet_op(_: MagicMock, config_path: str, variable: str):
    input_item = DataVibe("input_item", FAKE_TIME_RANGE, mapping(FAKE_GEOMETRY), [])

    op_tester = OpTester(config_path)
    op_tester.update_parameters({"variable": variable})
    output_data = op_tester.run(input_item=input_item)

    # Get op result
    output_name = "products"
    assert output_name in output_data
    output_product = output_data[output_name]
    assert isinstance(output_product, list)
    assert len(cast(List[ClimatologyLabProduct], output_data["products"])) == 2


@pytest.mark.parametrize("config_path", [TERRACLIMATE_CONFIG_PATH, GRIDMET_CONFIG_PATH])
def test_op_fails_invalid_variable(config_path: str):
    op_tester = OpTester(config_path)
    op_tester.update_parameters({"variable": INVALID_VARIABLE})
    with pytest.raises(ValueError):
        op_tester.run(input_item=[])
