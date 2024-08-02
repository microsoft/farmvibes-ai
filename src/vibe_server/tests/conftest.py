# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import asdict
from typing import Any, Dict

import pytest

from vibe_common.messaging import WorkMessage
from vibe_core.datamodel import RunConfig, RunDetails, RunStatus, SpatioTemporalJson
from vibe_dev.testing import anyio_backend
from vibe_dev.testing.fake_workflows_fixtures import fake_ops_dir, fake_workflows_dir
from vibe_dev.testing.workflow_fixtures import (
    SimpleStrData,
    SimpleStrDataType,
    workflow_execution_message,
    workflow_run_config,
)


@pytest.fixture
def run_config(workflow_execution_message: WorkMessage) -> Dict[str, Any]:
    run_id = workflow_execution_message.header.run_id
    spatio_temporal_json = {
        "end_date": "2019-02-03T00:00:00",
        "geojson": {
            "features": [
                {
                    "geometry": {
                        "coordinates": [
                            [
                                [-88.068487, 37.058836],
                                [-88.036059, 37.048687],
                                [-88.012895, 37.068984],
                                [-88.026622, 37.085711],
                                [-88.062482, 37.081461],
                                [-88.068487, 37.058836],
                            ]
                        ],
                        "type": "Polygon",
                    },
                    "type": "Feature",
                }
            ],
            "type": "FeatureCollection",
        },
        "start_date": "2019-02-02T00:00:00",
    }

    run_config = asdict(
        RunConfig(
            name="fake",
            workflow="fake",
            parameters=None,
            user_input=SpatioTemporalJson(**spatio_temporal_json),
            id=run_id,
            details=RunDetails(
                status=RunStatus.running, start_time=None, end_time=None, reason=None
            ),
            task_details={},
            spatio_temporal_json=None,
            output="",
        )
    )
    return run_config


__all__ = [
    "SimpleStrData",
    "SimpleStrDataType",
    "workflow_execution_message",
    "fake_ops_dir",
    "fake_workflows_dir",
    "workflow_run_config",
    "anyio_backend",
    "run_config",
]
