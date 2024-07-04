import pytest

from vibe_dev.testing.fake_workflows_fixtures import get_fake_workflow_path
from vibe_server.workflow.spec_parser import WorkflowParser
from vibe_server.workflow.spec_validator import WorkflowSpecValidator


def test_validator_fails_unused_parameter(fake_ops_dir: str, fake_workflows_dir: str):
    wf_path = get_fake_workflow_path("resolve_params")
    wf_dict = WorkflowParser._load_workflow(wf_path)
    spec = WorkflowParser.parse_dict(
        wf_dict, ops_dir=fake_ops_dir, workflows_dir=fake_workflows_dir
    )
    WorkflowSpecValidator.validate(spec)
    # Add unused param
    wf_dict["parameters"]["unused"] = None
    spec = WorkflowParser.parse_dict(
        wf_dict, ops_dir=fake_ops_dir, workflows_dir=fake_workflows_dir
    )
    with pytest.raises(ValueError):
        WorkflowSpecValidator._validate_parameter_references(spec)


def test_validator_fails_bad_ref(fake_ops_dir: str, fake_workflows_dir: str):
    wf_path = get_fake_workflow_path("resolve_params")
    wf_dict = WorkflowParser._load_workflow(wf_path)
    # Add invalid ref
    wf_dict["tasks"]["nested"]["parameters"]["overwrite"] = "@from(unexistent)"
    spec = WorkflowParser.parse_dict(wf_dict, fake_ops_dir, workflows_dir=fake_workflows_dir)
    with pytest.raises(ValueError):
        WorkflowSpecValidator._validate_parameter_references(spec)


def test_validator_fails_multiple_defaults(fake_ops_dir: str, fake_workflows_dir: str):
    wf_path = get_fake_workflow_path("resolve_nested_params_multiple_default")
    spec = WorkflowParser.parse(wf_path, ops_dir=fake_ops_dir, workflows_dir=fake_workflows_dir)
    with pytest.raises(ValueError):
        WorkflowSpecValidator.validate(spec)


def test_validator_fails_source_and_destination(fake_ops_dir: str, fake_workflows_dir: str):
    wf_path = get_fake_workflow_path("source_and_destination")
    spec = WorkflowParser.parse(wf_path, ops_dir=fake_ops_dir, workflows_dir=fake_workflows_dir)
    with pytest.raises(ValueError):
        WorkflowSpecValidator.validate(spec)
