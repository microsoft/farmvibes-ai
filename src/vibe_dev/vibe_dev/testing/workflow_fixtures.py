# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Type

import pytest
import yaml
from shapely import geometry as shpg

from vibe_common.messaging import (
    MessageHeader,
    MessageType,
    WorkflowExecutionContent,
    WorkMessage,
    build_work_message,
)
from vibe_common.schemas import EntryPointDict, OperationSpec
from vibe_core.data import DataVibe, TypeDictVibe
from vibe_core.data.core_types import BaseVibe
from vibe_core.datamodel import TaskDescription

from .fake_workflows_fixtures import get_fake_workflow_path


@dataclass
class SimpleStrDataType(BaseVibe):
    data: str


@pytest.fixture
def SimpleStrData() -> Type[SimpleStrDataType]:
    # A fixture that creates a type. Should this be in snake_case, or in CamelCase?
    # I went with CamelCase, as there is no way to make this pretty.
    return SimpleStrDataType


@pytest.fixture
def workflow_execution_message(SimpleStrData: Type[SimpleStrDataType]) -> WorkMessage:
    with open(get_fake_workflow_path("item_gather")) as f:
        wf_dict = yaml.safe_load(f)

    header = MessageHeader(
        type=MessageType.workflow_execution_request,
        run_id=uuid.uuid4(),
    )
    data = SimpleStrData("some fake data")
    content = WorkflowExecutionContent(
        name="fake_item_gather",
        input={
            "plain_input": {"data": data},
        },
        workflow=wf_dict,
    )
    return build_work_message(header=header, content=content)


@pytest.fixture
def simple_op_spec(SimpleStrData: Type[SimpleStrDataType], tmp_path: Path) -> OperationSpec:
    with open(tmp_path / "fake.py", "w") as fp:
        fp.write(
            """
from datetime import datetime
from vibe_core.data import DataVibe
from vibe_dev.testing.workflow_fixtures import SimpleStrDataType as SimpleStrData
def fake_callback(*args, **kwargs):
    def callback(**kwargs):
        out = {
            "processed_data": DataVibe(
                id="🍔",
                time_range=(datetime.now(), datetime.now()),
                geometry={
                    "type": "Point",
                    "coordinates": [0.0, 0.0],
                    "properties": {
                        "name": "🤭"
                    }
                },
                assets=[]
            ),
            "simple_str": SimpleStrData("🍔")
        }
        return out
    return callback

        """
        )
    return OperationSpec(
        name="fake",
        inputs_spec=TypeDictVibe(
            {  # type: ignore
                "plain_input": SimpleStrData,
                "list_input": List[SimpleStrData],
                "terravibes_input": DataVibe,
                "terravibes_list": List[DataVibe],
            }
        ),
        output_spec=TypeDictVibe({"processed_data": DataVibe, "simple_str": SimpleStrData}),
        parameters={},
        entrypoint=EntryPointDict(
            {"file": "fake.py", "callback_builder": "fake_callback"}  # type: ignore
        ),
        root_folder=str(tmp_path),
        description=TaskDescription(),
    )


@pytest.fixture
def workflow_run_config() -> Dict[str, Any]:
    return {
        "name": "fake workflow run",
        "user_input": {
            "start_date": "2021-02-02T00:00:00",
            "end_date": "2021-02-09T00:00:00",
            "geojson": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
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
                        },
                    }
                ],
            },
        },
        "workflow": "helloworld",
        "parameters": None,
    }


COORDS = (-55, -6)
TIME_RANGE = (datetime.now(), datetime.now())
THE_DATAVIBE = DataVibe(
    id="1",
    time_range=TIME_RANGE,
    geometry=shpg.mapping(shpg.Point(*COORDS).buffer(0.05, cap_style=3)),
    assets=[],
)
