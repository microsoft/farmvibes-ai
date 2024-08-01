# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import asdict

import pytest
import yaml

from vibe_dev.testing.fake_workflows_fixtures import get_fake_workflow_path
from vibe_server.workflow.spec_parser import WorkflowParser


@pytest.mark.parametrize("missing_field", WorkflowParser.required_fields)
def test_parser_fails_missing_field(missing_field: str, fake_ops_dir: str, fake_workflows_dir: str):
    wf_path = get_fake_workflow_path("resolve_params")
    with open(wf_path) as f:
        wf_dict = yaml.safe_load(f)
    del wf_dict[missing_field]
    with pytest.raises(ValueError):
        WorkflowParser.parse_dict(wf_dict, ops_dir=fake_ops_dir, workflows_dir=fake_workflows_dir)


def test_parser_fails_unknown_wf_field(fake_ops_dir: str, fake_workflows_dir: str):
    wf_path = get_fake_workflow_path("resolve_params")
    with open(wf_path) as f:
        wf_dict = yaml.safe_load(f)
    wf_dict["unknown"] = "🤦‍♂️"

    with pytest.raises(ValueError):
        WorkflowParser.parse_dict(wf_dict, ops_dir=fake_ops_dir, workflows_dir=fake_workflows_dir)


def test_parser_fails_unknown_task_field(fake_ops_dir: str, fake_workflows_dir: str):
    wf_path = get_fake_workflow_path("resolve_params")
    with open(wf_path) as f:
        wf_dict = yaml.safe_load(f)

    wf_dict["tasks"]["simple"]["unknown"] = "🤦‍♂"

    with pytest.raises(ValueError):
        WorkflowParser.parse_dict(wf_dict, ops_dir=fake_ops_dir, workflows_dir=fake_workflows_dir)


def test_parser_fills_optional_fields(fake_ops_dir: str, fake_workflows_dir: str):
    wf_path = get_fake_workflow_path("list_list")
    with open(wf_path) as f:
        wf_dict = yaml.safe_load(f)
    for field in WorkflowParser.optional_fields:
        wf_dict[field] = None
        spec = WorkflowParser.parse_dict(
            wf_dict, ops_dir=fake_ops_dir, workflows_dir=fake_workflows_dir
        )
        assert getattr(spec, field) is not None
        del wf_dict[field]


def test_parser_parameter_override(fake_ops_dir: str, fake_workflows_dir: str):
    spec = WorkflowParser.parse(
        get_fake_workflow_path("resolve_params"),
        ops_dir=fake_ops_dir,
        workflows_dir=fake_workflows_dir,
        parameters_override={"new": "override"},
    )
    assert spec.parameters["new"] == "override"


def test_parser_parameter_override_yaml_dict(fake_ops_dir: str, fake_workflows_dir: str):
    wf_path = get_fake_workflow_path("resolve_params")
    with open(wf_path) as f:
        wf_dict = yaml.safe_load(f)
    spec = WorkflowParser.parse_dict(
        wf_dict,
        ops_dir=fake_ops_dir,
        workflows_dir=fake_workflows_dir,
        parameters_override={"new": "override"},
    )
    assert spec.parameters["new"] == "override"


def test_parser_parameter_override_spec_dict(fake_ops_dir: str, fake_workflows_dir: str):
    spec = WorkflowParser.parse(
        get_fake_workflow_path("resolve_params"),
        ops_dir=fake_ops_dir,
        workflows_dir=fake_workflows_dir,
    )
    spec = WorkflowParser.parse_dict(
        asdict(spec),
        ops_dir=fake_ops_dir,
        workflows_dir=fake_workflows_dir,
        parameters_override={"new": "override"},
    )
    assert spec.parameters["new"] == "override"
