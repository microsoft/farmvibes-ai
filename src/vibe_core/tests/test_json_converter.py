# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from pydantic import parse_obj_as
from vibe_core.data.json_converter import dump_to_json
from vibe_core.datamodel import (
    RunConfig,
    RunConfigUser,
    RunDetails,
    RunStatus,
    SpatioTemporalJson,
    encode,
)


def test_run_config_serialization_does_not_redecorate():
    timestamp = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    run_id = UUID("12345678-1234-5678-1234-567812345678")
    area_json = {
        "start_date": timestamp.isoformat(),
        "end_date": timestamp.isoformat(),
        "geojson": {"type": "Point", "coordinates": [1.0, 2.0]},
    }
    output = {"task": {"value": 1}}
    encoded_output = encode(dump_to_json(output))
    run_data = {
        "name": "run",
        "workflow": {"name": "workflow"},
        "parameters": {"threshold": 1},
        "user_input": {
            "start_date": timestamp,
            "end_date": timestamp,
            "geojson": {"type": "Point", "coordinates": [1.0, 2.0]},
        },
        "id": run_id,
        "details": {"submission_time": timestamp, "status": RunStatus.done},
        "task_details": {
            "task": {"end_time": timestamp, "status": RunStatus.done},
        },
        "spatio_temporal_json": {
            "start_date": timestamp,
            "end_date": timestamp,
            "geojson": {"type": "Point", "coordinates": [1.0, 2.0]},
        },
        "output": encoded_output,
    }
    expected = {
        "name": "run",
        "workflow": {"name": "workflow"},
        "parameters": {"threshold": 1},
        "user_input": area_json,
        "id": str(run_id),
        "details": {
            "start_time": None,
            "submission_time": timestamp.isoformat(),
            "end_time": None,
            "reason": None,
            "status": "done",
            "subtasks": None,
        },
        "task_details": {
            "task": {
                "start_time": None,
                "submission_time": None,
                "end_time": timestamp.isoformat(),
                "reason": None,
                "status": "done",
                "subtasks": None,
            }
        },
        "spatio_temporal_json": area_json,
        "output": encoded_output,
    }
    initial_init = RunConfig.__init__
    initial_model = RunConfig.__dict__["__pydantic_model__"]
    restored = parse_obj_as(RunConfig, expected)

    for _ in range(2000):
        run = parse_obj_as(RunConfig, run_data)
        serialized = json.loads(dump_to_json(run))
        restored = parse_obj_as(RunConfig, serialized)
        assert serialized == expected

    assert RunConfig.__init__ is initial_init
    assert RunConfig.__dict__["__pydantic_model__"] is initial_model
    assert type(restored) is RunConfig
    assert isinstance(restored.id, UUID)
    assert restored.id == run_id
    assert type(restored.details) is RunDetails
    assert isinstance(restored.details.submission_time, datetime)
    assert restored.details.submission_time == timestamp
    assert type(restored.user_input) is SpatioTemporalJson
    assert type(restored.spatio_temporal_json) is SpatioTemporalJson
    assert type(restored.task_details["task"]) is RunDetails
    assert restored.output == encoded_output
    assert RunConfigUser.from_runconfig(restored).output == output


def test_undecorated_dataclass_serialization():
    @dataclass
    class PlainDataclass:
        timestamp: datetime
        identifier: UUID

    timestamp = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    identifier = UUID("12345678-1234-5678-1234-567812345678")
    assert json.loads(dump_to_json(PlainDataclass(timestamp, identifier))) == {
        "timestamp": timestamp.isoformat(),
        "identifier": str(identifier),
    }

    @dataclass
    class InheritedDataclass(RunDetails):
        label: str = "task"

    assert "__pydantic_model__" not in InheritedDataclass.__dict__
    assert json.loads(dump_to_json(InheritedDataclass()))["label"] == "task"
