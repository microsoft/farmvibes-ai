# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, cast
from unittest.mock import MagicMock, patch

import pytest
from shapely import geometry as shpg

from vibe_common.input_handlers import gen_stac_item_from_bounds
from vibe_core.data.core_types import BaseVibe, DataVibe, OpIOType
from vibe_core.data.rasters import CategoricalRaster, Raster
from vibe_core.data.utils import StacConverter, serialize_stac
from vibe_core.datamodel import SpatioTemporalJson
from vibe_dev.testing.fake_workflows_fixtures import get_fake_workflow_path
from vibe_server.workflow.input_handler import (
    build_args_for_workflow,
    patch_workflow_sources,
    validate_workflow_input,
)
from vibe_server.workflow.spec_parser import WorkflowParser
from vibe_server.workflow.workflow import Workflow


@pytest.fixture
def dummy_input():
    return {"dummy": 0, "another": "1"}


def test_build_workflow_args_spatiotemporal_input():
    geom = shpg.box(0, 0, 1, 1)
    geojson = {"type": "Feature", "geometry": shpg.mapping(geom)}
    start_date = datetime(2020, 1, 2)
    end_date = datetime(2020, 1, 3)
    user_input = SpatioTemporalJson(start_date, end_date, geojson)
    args = build_args_for_workflow(user_input, ["one_input"])
    assert args == {"one_input": gen_stac_item_from_bounds(geojson, start_date, end_date)}
    with pytest.raises(ValueError):
        build_args_for_workflow(user_input, ["1", "2"])


def test_build_workflow_args_single_source(dummy_input: Dict[str, Any]):
    args = build_args_for_workflow(dummy_input, ["one_input"])
    assert args == {"one_input": dummy_input}
    args = build_args_for_workflow({"one_input": dummy_input}, ["one_input"])
    assert args == {"one_input": dummy_input}


def test_build_workflow_args_multi_source(dummy_input: Dict[str, Any]):
    inputs = ["1", "2"]
    matching_input = {k: dummy_input for k in inputs}
    args = build_args_for_workflow(matching_input, inputs)
    assert args == matching_input


def test_build_workflow_args_missing_key_fails(dummy_input: Dict[str, Any]):
    inputs = ["1", "2"]
    with pytest.raises(ValueError):
        build_args_for_workflow(dummy_input, inputs)


def test_build_workflow_args_wrong_key_fails(dummy_input: Dict[str, Any]):
    inputs = ["1", "2"]
    bad_input = {k: dummy_input for k in ["1", "3"]}
    with pytest.raises(ValueError):
        build_args_for_workflow(bad_input, inputs)


def test_validate_wf_item_input():
    inputs_spec: Any = {"input": DataVibe}
    converter = StacConverter()
    geom = shpg.box(0, 0, 1, 1)
    now = datetime.now()
    x = DataVibe(id="1", time_range=(now, now), geometry=shpg.mapping(geom), assets=[])
    serial = serialize_stac(converter.to_stac_item(x))
    validate_workflow_input(cast(OpIOType, {"input": serial}), inputs_spec)


def test_validate_wf_input_subtype():
    inputs_spec: Any = {"input": Raster}
    converter = StacConverter()
    geom = shpg.box(0, 0, 1, 1)
    now = datetime.now()
    x = Raster(id="1", time_range=(now, now), geometry=shpg.mapping(geom), assets=[], bands={})
    serial = serialize_stac(converter.to_stac_item(x))
    validate_workflow_input(cast(OpIOType, {"input": serial}), inputs_spec)

    # More specific types are fine
    x = CategoricalRaster.clone_from(x, id="2", assets=[], categories=[])
    serial = serialize_stac(converter.to_stac_item(x))
    validate_workflow_input(cast(OpIOType, {"input": serial}), inputs_spec)

    # More generic types are not
    x = DataVibe.clone_from(x, id="3", assets=[])
    serial = serialize_stac(converter.to_stac_item(x))
    with pytest.raises(ValueError):
        validate_workflow_input(cast(OpIOType, {"input": serial}), inputs_spec)


def test_validate_wf_list_input():
    inputs_spec: Any = {"input": List[DataVibe]}
    converter = StacConverter()
    geom = shpg.box(0, 0, 1, 1)
    now = datetime.now()
    x = DataVibe(id="1", time_range=(now, now), geometry=shpg.mapping(geom), assets=[])
    serial = serialize_stac(converter.to_stac_item(x))
    validate_workflow_input(cast(OpIOType, {"input": [serial]}), inputs_spec)
    # Item is ok as well (will be converted to one item list)
    validate_workflow_input(cast(OpIOType, {"input": serial}), inputs_spec)


def test_validate_wf_base_input():
    @dataclass
    class A(BaseVibe):
        a: int

    inputs_spec: Any = {"input": List[A]}
    input = serialize_stac(StacConverter().to_stac_item(A(a=1)))
    other_input = copy.deepcopy(input)
    del other_input["properties"]["a"]
    other_input["properties"]["b"] = 1

    validate_workflow_input({"input": input}, inputs_spec)
    validate_workflow_input({"input": [input]}, inputs_spec)

    with pytest.raises(ValueError):
        validate_workflow_input({"input": other_input}, inputs_spec)

    with pytest.raises(ValueError):
        validate_workflow_input({"input": [other_input]}, inputs_spec)

    inputs_spec: Any = {"input": A}
    validate_workflow_input({"input": input}, inputs_spec)


def test_validate_wf_multi_source_input():
    inputs_spec: Any = {"input1": DataVibe, "input2": Raster}
    converter = StacConverter()
    geom = shpg.box(0, 0, 1, 1)
    now = datetime.now()
    x1 = DataVibe(id="1", time_range=(now, now), geometry=shpg.mapping(geom), assets=[])
    s1 = serialize_stac(converter.to_stac_item(x1))
    x2 = Raster.clone_from(x1, id="1", assets=[], bands={})
    s2 = serialize_stac(converter.to_stac_item(x2))
    x3 = CategoricalRaster.clone_from(x2, id="1", assets=[], categories=[])
    s3 = serialize_stac(converter.to_stac_item(x3))

    validate_workflow_input({"input1": s1, "input2": s2}, inputs_spec)
    validate_workflow_input({"input1": s1, "input2": s3}, inputs_spec)
    validate_workflow_input({"input1": s3, "input2": s2}, inputs_spec)

    with pytest.raises(ValueError):
        validate_workflow_input({"input1": s1, "input2": s1}, inputs_spec)


def test_workflow_source_patch(fake_ops_dir: str, fake_workflows_dir: str):
    workflow = Workflow.build(get_fake_workflow_path("item_item"), fake_ops_dir, fake_workflows_dir)
    assert workflow.inputs_spec == {"input": DataVibe}
    assert len(workflow.nodes) == 1
    assert len(workflow.edges) == 0
    old_source = workflow.source_mappings["input"][0]
    patch_workflow_sources({"input": []}, workflow)
    # We support list in the input
    assert workflow.inputs_spec == {"input": List[DataVibe]}
    # We add one fan-out node
    assert len(workflow.nodes) == 2
    # We add one edge from fan-out node to actual node
    assert len(workflow.edges) == 1
    # Our new edge should be from our node to the former source port
    edge = workflow.edges_from(workflow.index["input_fanout"])[0]
    destination = f"{edge[1].name}.{edge[2][1]}"
    assert destination == old_source


def test_workflow_source_patch_multiedge(fake_ops_dir: str, fake_workflows_dir: str):
    workflow = Workflow.build(
        get_fake_workflow_path("specific_source"), fake_ops_dir, fake_workflows_dir
    )
    assert workflow.inputs_spec == {"input": Raster}
    assert len(workflow.nodes) == 2
    assert len(workflow.edges) == 0
    old_sources = [s for s in workflow.source_mappings["input"]]
    patch_workflow_sources({"input": []}, workflow)
    # We support list in the input
    assert workflow.inputs_spec == {"input": List[Raster]}
    # We add one fan-out node
    assert len(workflow.nodes) == 3
    # We add one edge from fan-out node to each input port in the source (2)
    assert len(workflow.edges) == 2
    # Each new edge should be from our node to a former source port
    edges = workflow.edges_from(workflow.index["input_fanout"])
    destinations = [f"{edge[1].name}.{edge[2][1]}" for edge in edges]
    assert sorted(destinations) == sorted(old_sources)


def test_workflow_source_patch_fails_nested_fanout(fake_ops_dir: str, fake_workflows_dir: str):
    workflow = Workflow.build(
        get_fake_workflow_path("fan_out_and_in"), fake_ops_dir, fake_workflows_dir
    )
    with pytest.raises(ValueError):
        patch_workflow_sources({"input": []}, workflow)


@patch("vibe_server.workflow.input_handler.fan_out_workflow_source")
def test_workflow_source_patch_list_source(
    patch_mock: MagicMock, fake_ops_dir: str, fake_workflows_dir: str
):
    workflow = Workflow.build(get_fake_workflow_path("list_list"), fake_ops_dir, fake_workflows_dir)
    patch_workflow_sources({"input": []}, workflow)
    # Put something that is not a list
    patch_workflow_sources({"input": 0}, workflow)  # type: ignore
    patch_mock.assert_not_called()


def test_workflow_multi_source_patch(fake_ops_dir: str, fake_workflows_dir: str):
    wf_dict = {
        "name": "test",
        "sources": {
            "input1": ["t1.input"],
            "input2": ["t2.input"],
            "input3": ["t3.input"],
        },
        "sinks": {
            "output1": "t1.output",
            "output2": "t2.gather",
            "output3": "t3.raster",
        },
        "tasks": {
            "t1": {"workflow": "item_gather"},
            "t2": {"workflow": "fan_out_and_in"},
            "t3": {"workflow": "specific_source"},
        },
    }
    spec = WorkflowParser.parse_dict(
        wf_dict, ops_dir=fake_ops_dir, workflows_dir=fake_workflows_dir
    )
    workflow = Workflow(spec)
    unpatched_nodes = len(workflow.nodes)
    unpatched_edges = len(workflow.edges)
    assert workflow.inputs_spec == {"input1": DataVibe, "input2": DataVibe, "input3": Raster}
    with pytest.raises(ValueError):
        patch_workflow_sources({"input1": [], "input2": [], "input3": []}, workflow)
    # We patched the first and last ones, but reverted the second one
    assert workflow.inputs_spec == {
        "input1": List[DataVibe],
        "input2": DataVibe,
        "input3": List[Raster],
    }
    assert len(workflow.nodes) == unpatched_nodes + 2
    # t3 has a source that maps to two ports so it will create two edges
    assert len(workflow.edges) == unpatched_edges + 3
