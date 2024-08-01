# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

from vibe_common.schemas import OperationParser
from vibe_dev.testing.fake_workflows_fixtures import get_fake_workflow_path
from vibe_server.workflow.parameter import Parameter, ParameterResolver
from vibe_server.workflow.spec_parser import WorkflowParser


def test_parameter_defaults_from_child():
    p_root = Parameter("root", "root", None, None, None)
    p_child = Parameter("child", "task", "@from(root)", 0, "child description")
    p_root.add_child(p_child)
    assert p_root.default == p_child.default
    assert p_root.description == p_child.description
    p_root._default = "set default"
    assert p_root.default == p_root._default
    assert p_root.description == p_child.description
    p_root._description = "set desc"
    assert p_root.default == p_root._default
    assert p_root.description == p_root._description
    p_root._default = None
    assert p_root.default == p_child.default
    assert p_root.description == p_root._description


def test_parameter_two_children():
    p_root = Parameter("root", "root", None, None, None)
    p_child = Parameter("child", "task", "@from(root)", 0, "child1 description")
    p_child2 = Parameter("child2", "task2", "@from(root)", 1, "child2 description")
    p_root.add_child(p_child)
    p_root.add_child(p_child2)
    assert p_root.default == (p_child.default, p_child2.default)
    assert p_root.description == (p_child.description, p_child2.description)


def test_parameter_two_children_same_definition():
    p_root = Parameter("root", "root", None, None, None)
    p_child = Parameter("child", "task", "@from(root)", 0, "child description")
    p_child2 = Parameter("child2", "task2", "@from(root)", 0, "child description")
    p_root.add_child(p_child)
    p_root.add_child(p_child2)
    assert p_root.default == p_child.default == p_child2.default
    assert p_root.description == p_child.description == p_child2.description


def test_parameter_children_handle_none():
    p_root = Parameter("root", "root", None, None, None)
    p_child = Parameter("child", "task", "@from(root)", 0, "child1 description")
    p_child2 = Parameter("child2", "task2", "@from(root)", None, None)
    p_root.add_child(p_child)
    p_root.add_child(p_child2)
    # For parameters, we don't discard None!
    assert p_root.default == (p_child.default, p_child2.default)
    # For descriptions, we ignore None from child2
    assert p_root.description == p_child.description
    p_child3 = Parameter("child", "task", "@from(root)", 2, "child3 description")
    p_root.add_child(p_child3)
    assert p_root.default == (p_child.default, p_child2.default, p_child3.default)
    assert p_root.description == (p_child.description, p_child3.description)


def test_get_op_params(fake_ops_dir: str):
    resolver = ParameterResolver("", "")
    op_spec = OperationParser.parse(
        os.path.join(fake_ops_dir, "fake", "simple_parameter.yaml"), {"overwrite": "over"}
    )
    params = {p.name: p for p in resolver._get_op_params(op_spec, "task")}
    assert len(params) == 2
    assert params["keep"]._value == "kept"
    assert params["keep"].default == "kept"
    assert params["keep"].description is None

    assert params["overwrite"]._value == "over"
    assert params["overwrite"].default == "kept"
    assert params["overwrite"].description is None


def test_get_op_params_nested(fake_ops_dir: str):
    resolver = ParameterResolver("", "")
    op_spec = OperationParser.parse(
        os.path.join(fake_ops_dir, "fake", "nested_parameters.yaml"),
        {"nested": {"overwrite": "over nested"}},
    )
    params = {p.name: p for p in resolver._get_op_params(op_spec, "task")}
    assert len(params) == 3
    param = params["overwrite"]
    assert param._value == param.default == "kept"
    assert param.description == "param named overwrite"

    param = params["nested.overwrite"]
    assert param._value == "over nested"
    assert param.default == "kept nested"
    assert param.description == "nested overwrite"


def test_resolve_params(fake_ops_dir: str, fake_workflows_dir: str):
    wf_path = get_fake_workflow_path("resolve_nested_params_multiple_default")
    wf_spec = WorkflowParser.parse(wf_path, ops_dir=fake_ops_dir, workflows_dir=fake_workflows_dir)
    resolver = ParameterResolver(fake_workflows_dir, fake_ops_dir)
    params = resolver.resolve(wf_spec)
    assert len(params) == 2
    param = params["new"]
    assert param.default == ("kept", "overwritten")
    assert param._value is None
    assert len(param.childs) == 2
    assert sorted([p.name for p in param.childs]) == ["new", "overwrite"]

    param = params["new_nested"]
    assert param.default == "overwritten nested"
    assert param._value is None
    assert len(param.childs) == 1
    assert param.description == "nested overwrite"


def test_resolve_only_description(fake_ops_dir: str, fake_workflows_dir: str):
    wf_path = get_fake_workflow_path("resolve_params")
    wf_spec = WorkflowParser.parse(wf_path, ops_dir=fake_ops_dir, workflows_dir=fake_workflows_dir)
    resolver = ParameterResolver(fake_workflows_dir, fake_ops_dir)
    params = resolver.resolve(wf_spec)
    param = params["new_nested"]
    # We don't get default from child
    assert param.default == param._value == "overwritten nested"
    # But we do get description
    assert param._description is None
    assert param.description == "nested overwrite"
