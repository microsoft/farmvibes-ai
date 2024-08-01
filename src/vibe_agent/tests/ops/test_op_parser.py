# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from typing import Any, Dict

from vibe_agent.ops import OperationParser, OperationSpec
from vibe_core.file_utils import write_yaml


def compare_spec_yaml(spec: OperationSpec, op_yaml: Dict[str, Any], root_folder: str):
    assert spec.dependencies == op_yaml.get("dependencies", {})
    assert spec.version == op_yaml.get("version", "1.0")
    assert spec.parameters == op_yaml["parameters"]
    assert spec.name == op_yaml["name"]
    assert spec.root_folder == root_folder
    assert spec.entrypoint["file"] == op_yaml["entrypoint"]["file"]
    assert spec.entrypoint["callback_builder"] == op_yaml["entrypoint"]["callback_builder"]
    assert op_yaml["inputs"].keys() == spec.inputs_spec.keys()


def test_parser_only_required(tmpdir: str, op_yaml: Dict[str, Any]):
    op_yaml_file = os.path.join(tmpdir, "fake.yaml")
    write_yaml(op_yaml_file, op_yaml)
    spec = OperationParser().parse(op_yaml_file)
    compare_spec_yaml(spec, op_yaml, tmpdir)


def test_parser_version(tmpdir: str, op_yaml: Dict[str, Any]):
    op_yaml_file = os.path.join(tmpdir, "fake.yaml")
    op_yaml["version"] = "2.5"
    write_yaml(op_yaml_file, op_yaml)
    spec = OperationParser().parse(op_yaml_file)
    compare_spec_yaml(spec, op_yaml, tmpdir)


def test_parser_dependencies(tmpdir: str, op_yaml: Dict[str, Any]):
    op_yaml_file = os.path.join(tmpdir, "fake.yaml")
    op_yaml["dependencies"] = {"parameters": ["fake_param"]}
    write_yaml(op_yaml_file, op_yaml)
    spec = OperationParser().parse(op_yaml_file)
    compare_spec_yaml(spec, op_yaml, tmpdir)


def test_parser_empty_fields(tmpdir: str, op_yaml: Dict[str, Any]):
    op_yaml_file = os.path.join(tmpdir, "fake.yaml")
    op_yaml["dependencies"] = None
    op_yaml["version"] = None
    op_yaml["parameters"] = None
    write_yaml(op_yaml_file, op_yaml)
    spec = OperationParser().parse(op_yaml_file)
    assert spec.parameters == {}
    assert spec.dependencies == {}
    assert spec.version == "1.0"
