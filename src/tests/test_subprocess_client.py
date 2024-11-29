# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from datetime import datetime, timezone
from typing import Tuple
from unittest.mock import Mock, patch

import pytest
from shapely.geometry import Polygon

from vibe_core.datamodel import RunStatus
from vibe_dev.client.subprocess_client import SubprocessClient, get_default_subprocess_client

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def input_polygon() -> Polygon:
    polygon_coords = [
        (-88.062073563448919, 37.081397673802059),
        (-88.026349330507315, 37.085463858128762),
        (-88.026349330507315, 37.085463858128762),
        (-88.012445388773259, 37.069230099135126),
        (-88.035931592028305, 37.048441375086092),
        (-88.068120429075847, 37.058833638440767),
        (-88.062073563448919, 37.081397673802059),
    ]

    return Polygon(polygon_coords)


@pytest.fixture
def workflow_name() -> str:
    return "helloworld"


@pytest.fixture
def workflow_path() -> str:
    return os.path.join(HERE, "..", "..", "workflows", "helloworld.yaml")


@pytest.fixture
def time_range() -> Tuple[datetime, datetime]:
    return (
        datetime(year=2021, month=2, day=1, tzinfo=timezone.utc),
        datetime(year=2021, month=2, day=11, tzinfo=timezone.utc),
    )


@patch("vibe_agent.worker.Worker.is_workflow_complete", return_value=False)
@pytest.mark.anyio
async def test_local_client_with_workflow_name(
    _: Mock,
    input_polygon: Polygon,
    workflow_name: str,
    tmp_path: str,
    time_range: Tuple[datetime, datetime],
    capsys,  # type: ignore
):
    client: SubprocessClient = get_default_subprocess_client(tmp_path)
    with capsys.disabled():
        output = await client.run(workflow_name, input_polygon, time_range)
    assert output.status == RunStatus.done


@patch("vibe_agent.worker.Worker.is_workflow_complete", return_value=False)
@pytest.mark.anyio
async def test_local_client_with_workflow_path(
    _: Mock,
    input_polygon: Polygon,
    workflow_path: str,
    tmp_path: str,
    time_range: Tuple[datetime, datetime],
    capsys,  # type: ignore
):
    client: SubprocessClient = get_default_subprocess_client(tmp_path)
    with capsys.disabled():
        output = await client.run(workflow_path, input_polygon, time_range)
    assert output.status == RunStatus.done
