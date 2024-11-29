# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import shutil
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
import yaml

from vibe_agent.ops import OperationFactoryConfig
from vibe_agent.storage import LocalFileAssetManagerConfig, LocalStorageConfig
from vibe_common.secret_provider import AzureSecretProviderConfig
from vibe_core.data.core_types import BaseVibe, DataVibe, OpIOType
from vibe_core.data.utils import StacConverter, get_base_type, serialize_stac
from vibe_dev.local_runner import LocalWorkflowRunner
from vibe_dev.testing.fake_workflows_fixtures import FakeType, get_fake_workflow_path
from vibe_server.workflow import list_workflows
from vibe_server.workflow.description_validator import WorkflowDescriptionValidator
from vibe_server.workflow.runner import (
    NoOpStateChange,
    WorkflowCallback,
    WorkflowChange,
    WorkflowRunner,
)
from vibe_server.workflow.runner.task_io_handler import WorkflowIOHandler
from vibe_server.workflow.spec_parser import WorkflowParser, get_workflow_dir
from vibe_server.workflow.spec_parser import parse_edge_string as pes
from vibe_server.workflow.workflow import Workflow, load_workflow_by_name

HERE = os.path.dirname(os.path.abspath(__file__))


def serialize(base: BaseVibe):
    return serialize_stac(StacConverter().to_stac_item(base))  # type: ignore


def gen_local_runner(
    storage_spec: Any,
    workflow_path: str,
    fake_ops_path: str,
    workflows_path: str,
    callback: WorkflowCallback = NoOpStateChange,
) -> WorkflowRunner:
    factory_spec = OperationFactoryConfig(storage_spec, AzureSecretProviderConfig())
    workflow = Workflow.build(workflow_path, fake_ops_path, workflows_path)
    io_mapper = WorkflowIOHandler(workflow)
    return LocalWorkflowRunner.build(
        workflow,
        factory_spec=factory_spec,
        io_mapper=io_mapper,
        update_state_callback=callback,
        max_tries=5,
    )


def build_workflow_runner(
    tmp_path: Path,
    workflow_path: str,
    fake_ops_path: str,
    workflows_path: str,
    callback: WorkflowCallback = NoOpStateChange,
) -> WorkflowRunner:
    tmp_asset_path = os.path.join(str(tmp_path), "assets")
    storage_spec = LocalStorageConfig(
        local_path=str(tmp_path), asset_manager=LocalFileAssetManagerConfig(tmp_asset_path)
    )
    return gen_local_runner(
        storage_spec, workflow_path, fake_ops_path, workflows_path, callback=callback
    )


@pytest.mark.parametrize("workflow_name", list_workflows())
def test_workflows_load(workflow_name: str):
    workflow = load_workflow_by_name(workflow_name)
    assert not workflow.has_cycle()


@pytest.mark.parametrize(
    "workflow_name", [wf_name for wf_name in list_workflows() if not wf_name.startswith("private/")]
)
def test_workflows_description(workflow_name: str):
    workflow_dir = get_workflow_dir()
    workflow_path = os.path.join(workflow_dir, f"{workflow_name}.yaml")
    workflow_spec = WorkflowParser.parse(workflow_path)
    WorkflowDescriptionValidator.validate(workflow_spec)


@pytest.mark.parametrize("workflow_name", list_workflows())
def test_list_workflows_schema_generation(workflow_name: str):
    workflow = load_workflow_by_name(workflow_name)
    ret: Dict[str, Any] = {
        k: get_base_type(v).schema()
        for k, v in workflow.inputs_spec.items()  # type: ignore
    }
    assert ret


def strip_edges_and_nodes_from_workflow(
    tmp_path: Path,
    workflow_path: str,
    fake_ops_path: str,
    workflows_path: str,
    strip_sinks: bool = False,
    tasks_to_keep: int = 1,
    del_edges: bool = False,
) -> WorkflowRunner:
    base = WorkflowParser.parse(workflow_path, fake_ops_path, workflows_path)

    if len(base.tasks) > tasks_to_keep:
        must_exist = [t for i, t in enumerate(base.tasks.keys()) if i < tasks_to_keep]
        base.tasks = {m: base.tasks[m] for m in must_exist}
        base.sinks = {e.origin: e.origin for e in base.edges if pes(e.origin)[0] in must_exist}
        base.edges = []
        base.sources = {k: v for i, (k, v) in enumerate(base.sources.items()) if i < 1}

    if strip_sinks:
        base.sinks = {}

    if del_edges:
        base.edges = []  # type: ignore

    tasks = {k: v.to_dict() for k, v in base.tasks.items()}
    base = asdict(base)
    base["tasks"] = tasks

    tmp = tempfile.NamedTemporaryFile("w", delete=False)
    yaml.dump(base, tmp)  # type: ignore
    tmp.close()

    try:
        return build_workflow_runner(tmp_path, tmp.name, fake_ops_path, workflows_path)
    finally:
        os.unlink(tmp.name)


def test_no_sinks_workflow(
    tmp_path: Path,
    fake_ops_dir: str,
    fake_workflows_dir: str,
):
    with pytest.raises(ValueError):
        strip_edges_and_nodes_from_workflow(
            tmp_path,
            get_fake_workflow_path("nested_workflow"),
            fake_ops_dir,
            fake_workflows_dir,
            True,
        )


def test_degenerate_workflow(tmp_path: Path, fake_ops_dir: str, fake_workflows_dir: str):
    with pytest.raises(ValueError):
        # For the reader that might be asking what is going on here,
        # we will end up with a two-node workflow that only has a
        # single source. The idea of supporting "single" operation
        # workflows is that all operations are sources and sinks.
        # So, if that's not the case, then edges are required.
        strip_edges_and_nodes_from_workflow(
            tmp_path,
            get_fake_workflow_path("nested_workflow"),
            fake_ops_dir,
            fake_workflows_dir,
            tasks_to_keep=2,
            del_edges=True,
        )


@pytest.mark.anyio
async def test_arbitrary_input(
    tmp_path: Path,
    fake_ops_dir: str,
    fake_workflows_dir: str,
):
    runner = build_workflow_runner(
        tmp_path, get_fake_workflow_path("str_input"), fake_ops_dir, fake_workflows_dir
    )
    user_input = FakeType("fake workflow execution")
    out = await runner.run({k: serialize(user_input) for k in runner.workflow.inputs_spec})
    for outname in runner.workflow.output_spec:
        assert outname in out


@pytest.mark.parametrize("workflow_name", ["nested_workflow", "workflow_inception"])
@pytest.mark.anyio
async def test_composable_workflow(
    workflow_name: str,
    tmp_path: Path,
    fake_ops_dir: str,
    fake_workflows_dir: str,
):
    user_input = FakeType("fake workflow execution")

    runner = build_workflow_runner(
        tmp_path, get_fake_workflow_path(workflow_name), fake_ops_dir, fake_workflows_dir
    )
    out = await runner.run({k: serialize(user_input) for k in runner.workflow.inputs_spec})
    for outname in runner.workflow.output_spec:
        assert outname in out


@pytest.mark.anyio
async def test_ordered_times_in_workflow(
    tmp_path: Path,
    fake_ops_dir: str,
    fake_workflows_dir: str,
):
    state: Dict[str, Tuple[WorkflowChange, datetime]] = {}

    runner = build_workflow_runner(
        tmp_path, get_fake_workflow_path("nested_workflow"), fake_ops_dir, fake_workflows_dir
    )
    await runner.run({k: serialize(FakeType("test")) for k in runner.workflow.inputs_spec})

    previous = None
    for task in (t for t in state.keys() if t.startswith("t")):
        if previous is None:
            previous = state[task]
            continue
        assert previous[-1] < state[task][-1]


@pytest.mark.anyio
async def test_fan_out_single_element(tmp_path: Path, fake_ops_dir: str, fake_workflows_dir: str):
    spec = WorkflowParser.parse(
        get_fake_workflow_path("fan_out_and_in"), fake_ops_dir, fake_workflows_dir
    )
    tmp_asset_path = os.path.join(str(tmp_path), "assets")
    storage_spec = LocalStorageConfig(
        local_path=str(tmp_path), asset_manager=LocalFileAssetManagerConfig(tmp_asset_path)
    )
    factory_spec = OperationFactoryConfig(storage_spec, AzureSecretProviderConfig())
    for num_items in (1, 5):
        spec.tasks["to_list"].parameters["num_items"] = num_items
        workflow = Workflow(spec)
        io_mapper = WorkflowIOHandler(workflow)
        runner = LocalWorkflowRunner.build(
            workflow,
            io_mapper=io_mapper,
            factory_spec=factory_spec,
        )
        converter = StacConverter()
        x = DataVibe(
            "input",
            time_range=(datetime.now(), datetime.now()),
            geometry={"type": "Point", "coordinates": [0.0, 0.0]},
            assets=[],
        )
        out = await runner.run({"input": serialize_stac(converter.to_stac_item(x))})
        shutil.rmtree(tmp_path)  # Delete the cache
        assert all(len(o) == num_items for o in out.values())


@pytest.mark.anyio
async def test_gather_not_parallel(tmp_path: Path, fake_ops_dir: str, fake_workflows_dir: str):
    runner = build_workflow_runner(
        tmp_path, get_fake_workflow_path("item_gather"), fake_ops_dir, fake_workflows_dir
    )
    converter = StacConverter()
    x = DataVibe(
        "input",
        time_range=(datetime.now(), datetime.now()),
        geometry={"type": "Point", "coordinates": [0.0, 0.0]},
        assets=[],
    )
    out = await runner.run(
        {k: serialize_stac(converter.to_stac_item(x)) for k in runner.workflow.inputs_spec}
    )
    assert len(out) == 1


# TODO: Restore "remote" storage_spec after fixing CosmosDB permissions
@pytest.mark.parametrize("storage_spec", ["local"], indirect=True)
@pytest.mark.anyio
async def test_op_run_race_condition(storage_spec: Any, fake_ops_dir: str, fake_workflows_dir: str):
    runner = gen_local_runner(
        storage_spec, get_fake_workflow_path("workflow_inception"), fake_ops_dir, fake_workflows_dir
    )
    user_input = FakeType("fake workflow execution")
    await runner.run({k: serialize(user_input) for k in runner.workflow.inputs_spec})


@pytest.mark.parametrize("edges", [None, []])
def test_parser_loads_workflow_with_no_edges(
    edges: List[Optional[List[Any]]], fake_ops_dir: str, fake_workflows_dir: str
) -> None:
    workflow_dict = WorkflowParser._load_workflow(get_fake_workflow_path("fan_out_and_in"))
    workflow_dict["edges"] = edges
    WorkflowParser.parse_dict(workflow_dict, fake_ops_dir, fake_workflows_dir)


@pytest.mark.anyio
async def test_running_workflow_with_basevibe_edges(
    tmp_path: Path,
    fake_ops_dir: str,  # noqa
    fake_workflows_dir: str,  # noqa
    SimpleStrData: Any,
):
    data = StacConverter().to_stac_item(SimpleStrData("🍔"))  # type: ignore
    wf_input: OpIOType = {"input": serialize_stac(data)}

    tmp_asset_path = os.path.join(str(tmp_path), "assets")
    storage_spec = LocalStorageConfig(
        local_path=str(tmp_path), asset_manager=LocalFileAssetManagerConfig(tmp_asset_path)
    )

    runner = gen_local_runner(
        storage_spec, get_fake_workflow_path("base_base"), fake_ops_dir, fake_workflows_dir
    )
    out = await runner.run(wf_input)
    assert out
