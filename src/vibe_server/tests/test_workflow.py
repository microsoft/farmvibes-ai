import os
from typing import List

import pytest

from vibe_core.data.core_types import DataVibe
from vibe_core.data.rasters import Raster
from vibe_core.data.utils import is_vibe_list
from vibe_dev.testing.fake_workflows_fixtures import get_fake_workflow_path
from vibe_server.workflow.spec_parser import WorkflowParser, WorkflowSpec, WorkflowSpecEdge
from vibe_server.workflow.workflow import EdgeType, Workflow

HERE = os.path.dirname(os.path.abspath(__file__))


def test_workflow_parameters(
    fake_ops_dir: str,
    fake_workflows_dir: str,
):
    workflow = Workflow.build(
        get_fake_workflow_path("task_params"), fake_ops_dir, fake_workflows_dir
    )
    assert workflow["parameterizable"].parameters["fake_param"] == 3  # type: ignore
    assert workflow["parameterizable"].parameters["fake_another_param"] == {  # type: ignore
        "fake_nested": 2,
        "fake_nested_too": 3,
    }


def test_workflow_nested_parameters(
    fake_ops_dir: str,
    fake_workflows_dir: str,
):
    workflow = Workflow.build(
        get_fake_workflow_path("nested_task_params"), fake_ops_dir, fake_workflows_dir
    )
    assert workflow["parameterizable"].parameters["fake_param"] == 1  # type: ignore
    assert workflow["parameterizable"].parameters["fake_another_param"] == {  # type: ignore
        "fake_nested": 2,
        "fake_nested_too": 4,
    }


def test_workflow_unknown_parameter(
    fake_ops_dir: str,
    fake_workflows_dir: str,
):
    with pytest.raises(ValueError):
        Workflow.build(
            get_fake_workflow_path("unknown_task_params"), fake_ops_dir, fake_workflows_dir
        )


def test_misconfigured_workflow(
    fake_ops_dir: str,
    fake_workflows_dir: str,
):
    with pytest.raises(ValueError):
        Workflow.build(get_fake_workflow_path("missing_edge"), fake_ops_dir, fake_workflows_dir)


def test_fan_out_fan_in(
    fake_ops_dir: str,
    fake_workflows_dir: str,
):
    # Tests whether we support workflows with nodes
    # from List[DataVibe] <-> [DataVibe]
    Workflow.build(get_fake_workflow_path("fan_out_and_in"), fake_ops_dir, fake_workflows_dir)


def test_nested_fan_out_fails(
    fake_ops_dir: str,
    fake_workflows_dir: str,
):
    with pytest.raises(ValueError):
        Workflow.build(get_fake_workflow_path("nested_fan_out"), fake_ops_dir, fake_workflows_dir)


@pytest.mark.parametrize(
    "workflow_name",
    ["single_and_parallel", "gather_and_parallel", "gather_and_parallel_input_gather_output"],
)
def test_parallelism_two_edge_types(
    workflow_name: str,
    fake_ops_dir: str,
    fake_workflows_dir: str,
):
    workflow_path = get_fake_workflow_path(workflow_name)

    workflow_spec: WorkflowSpec = WorkflowParser.parse(
        workflow_path, fake_ops_dir, fake_workflows_dir
    )
    workflow = Workflow(workflow_spec)
    edge = workflow.edges_from(workflow.index["two_types"])[0]
    correct_type = EdgeType.gather if "gather_output" in workflow_name else EdgeType.parallel
    assert edge[-1].type == correct_type


def test_gather_not_parallel(
    fake_ops_dir: str,
    fake_workflows_dir: str,
):
    workflow = Workflow.build(
        get_fake_workflow_path("item_gather"), fake_ops_dir, fake_workflows_dir
    )
    assert workflow.edges_from(workflow.index["item"])[0][-1].type == EdgeType.gather


def test_loading_inheritance_works(
    fake_ops_dir: str,
    fake_workflows_dir: str,
):
    workflow = Workflow.build(
        get_fake_workflow_path("inheritance"), fake_ops_dir, fake_workflows_dir
    )
    assert not is_vibe_list(workflow["inherit_item"].output_spec["processed_data"])
    assert is_vibe_list(workflow["inherit_list"].output_spec["processed_data"])


def test_loading_missing_inheritance_fails(
    fake_ops_dir: str,
    fake_workflows_dir: str,
):
    with pytest.raises(ValueError):
        Workflow.build(
            get_fake_workflow_path("missing_inheritance"), fake_ops_dir, fake_workflows_dir
        )


def test_loading_multi_level_inheritance_works(
    fake_ops_dir: str,
    fake_workflows_dir: str,
):
    workflow = Workflow.build(
        get_fake_workflow_path("two_level_inheritance"), fake_ops_dir, fake_workflows_dir
    )
    assert workflow["direct_inherit"].output_spec["processed_data"] is DataVibe
    assert workflow["indirect_inherit"].output_spec["processed_data"] is DataVibe


def test_inheritance_before_fanout(
    fake_ops_dir: str,
    fake_workflows_dir: str,
):
    workflow = Workflow.build(
        get_fake_workflow_path("inheritance_before_fan_out"), fake_ops_dir, fake_workflows_dir
    )

    assert workflow["inherit_list"].output_spec["processed_data"] is List[DataVibe]
    assert list(workflow.edges_from(workflow.index["inherit_list"]))[0][-1].type == EdgeType.scatter


def test_inheritance_after_fanout(
    fake_ops_dir: str,
    fake_workflows_dir: str,
):
    workflow = Workflow.build(
        get_fake_workflow_path("inheritance_after_fan_out"), fake_ops_dir, fake_workflows_dir
    )

    assert workflow["scatter_inherit"].output_spec["processed_data"] is DataVibe
    assert list(workflow.edges_from(workflow.index["list"]))[0][-1].type == EdgeType.scatter
    assert (
        list(workflow.edges_from(workflow.index["scatter_inherit"]))[0][-1].type
        == EdgeType.parallel
    )


def test_inheritance_source(
    fake_ops_dir: str,
    fake_workflows_dir: str,
):
    workflow = Workflow.build(
        get_fake_workflow_path("inheritance_from_source"), fake_ops_dir, fake_workflows_dir
    )

    assert workflow["inherit_raster"].output_spec["processed_data"] is Raster
    assert workflow["inherit_source"].output_spec["processed_data"] is DataVibe


def test_cycle_disconnected_components_detection(
    fake_ops_dir: str,
    fake_workflows_dir: str,
):
    workflow_path = get_fake_workflow_path("three_ops")

    workflow_spec: WorkflowSpec = WorkflowParser.parse(
        workflow_path, fake_ops_dir, fake_workflows_dir
    )
    for origin, destination in zip(
        ("second.processed_data", "third.processed_data", "third.processed_data"),
        ("first.user_data", "second.user_data", "third.user_data"),
    ):
        edge: WorkflowSpecEdge = WorkflowSpecEdge(origin=origin, destination=[destination])
        workflow_spec.edges.append(edge)

        with pytest.raises(ValueError):
            Workflow(workflow_spec)

        workflow_spec.edges.pop()


def test_parameter_resolution(
    fake_ops_dir: str,
    fake_workflows_dir: str,
):
    workflow_path = get_fake_workflow_path("resolve_params")

    workflow = Workflow.build(workflow_path, fake_ops_dir, fake_workflows_dir)
    assert workflow["simple"].parameters["keep"] == "kept"
    assert workflow["simple"].parameters["overwrite"] == "overwritten"
    assert workflow["nested"].parameters["overwrite"] == "overwritten"
    assert workflow["nested"].parameters["nested"]["keep"] == "kept nested"
    assert workflow["nested"].parameters["nested"]["overwrite"] == "overwritten nested"


def test_nested_workflow_parameter_resolution(
    fake_ops_dir: str,
    fake_workflows_dir: str,
):
    workflow_path = get_fake_workflow_path("resolve_nested_params")

    workflow = Workflow.build(workflow_path, fake_ops_dir, fake_workflows_dir)
    assert workflow["simple"].parameters["keep"] == "kept"
    assert workflow["simple"].parameters["overwrite"] == "overwritten"
    assert workflow["nested.simple"].parameters["overwrite"] == "overwritten"
    assert workflow["nested.nested"].parameters["overwrite"] == "overwritten"
    assert workflow["nested.nested"].parameters["nested"]["keep"] == "kept nested"
    assert workflow["nested.nested"].parameters["nested"]["overwrite"] == "overwritten nested"


def test_workflow_parameter_resolution_default_values(fake_ops_dir: str, fake_workflows_dir: str):
    workflow_path = get_fake_workflow_path("resolve_nested_params_default")

    workflow = Workflow.build(workflow_path, fake_ops_dir, fake_workflows_dir)
    assert workflow["simple"].parameters["keep"] == "kept"
    # Default value for the op in 'overwrite' is "kept"
    assert workflow["simple"].parameters["overwrite"] == "kept"
    assert workflow["nested.simple"].parameters["overwrite"] == "overwritten"
    assert workflow["nested.nested"].parameters["overwrite"] == "overwritten"
    assert workflow["nested.nested"].parameters["nested"]["keep"] == "kept nested"
    # Default value for the op in 'overwrite' is kept,
    # but default for the workflow containing it is 'overwritten nested'
    assert workflow["nested.nested"].parameters["nested"]["overwrite"] == "overwritten nested"


@pytest.mark.parametrize("invalid", ["", "inexistent"])
def test_workflow_parameter_resolution_invalid_ref(
    fake_ops_dir: str, fake_workflows_dir: str, invalid: str
):
    workflow_path = get_fake_workflow_path("resolve_params")

    spec = WorkflowParser.parse(workflow_path, fake_ops_dir, fake_workflows_dir)
    spec.tasks["simple"].parameters["overwrite"] = f"@from({invalid})"
    with pytest.raises(ValueError):
        Workflow(spec)


def test_workflow_missing_source(fake_ops_dir: str, fake_workflows_dir: str):
    workflow_path = get_fake_workflow_path("bad_source")
    with pytest.raises(ValueError):
        Workflow.build(
            workflow_path, ops_base_dir=fake_ops_dir, workflow_base_dir=fake_workflows_dir
        )


def test_workflow_missing_sink(fake_ops_dir: str, fake_workflows_dir: str):
    workflow_path = get_fake_workflow_path("bad_sink")
    with pytest.raises(ValueError):
        Workflow.build(
            workflow_path, ops_base_dir=fake_ops_dir, workflow_base_dir=fake_workflows_dir
        )


def test_most_specific_source_type(fake_ops_dir: str, fake_workflows_dir: str):
    workflow_path = get_fake_workflow_path("specific_source")
    wf = Workflow.build(
        workflow_path, ops_base_dir=fake_ops_dir, workflow_base_dir=fake_workflows_dir
    )
    assert wf.inputs_spec["input"] is Raster


def test_item_list_source_type(fake_ops_dir: str, fake_workflows_dir: str):
    workflow_path = get_fake_workflow_path("specific_source_item_list")
    wf = Workflow.build(
        workflow_path, ops_base_dir=fake_ops_dir, workflow_base_dir=fake_workflows_dir
    )
    assert wf.inputs_spec["input"] is Raster


def test_list_list_source_type(fake_ops_dir: str, fake_workflows_dir: str):
    workflow_path = get_fake_workflow_path("specific_source_list_list")
    wf = Workflow.build(
        workflow_path, ops_base_dir=fake_ops_dir, workflow_base_dir=fake_workflows_dir
    )
    assert wf.inputs_spec["input"] is List[Raster]


def test_incompatible_sources_fails(fake_ops_dir: str, fake_workflows_dir: str):
    workflow_path = get_fake_workflow_path("incompatible_source")
    with pytest.raises(ValueError):
        Workflow.build(
            workflow_path, ops_base_dir=fake_ops_dir, workflow_base_dir=fake_workflows_dir
        )
