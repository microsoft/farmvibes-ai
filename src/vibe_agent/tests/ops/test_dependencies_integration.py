# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from vibe_agent.ops import EntryPointDict, OperationDependencyResolver, OperationSpec
from vibe_core.data import DataVibe, TypeDictVibe
from vibe_core.datamodel import TaskDescription


@pytest.fixture
def operation_spec():
    return OperationSpec(
        name="fake",
        inputs_spec=TypeDictVibe({"vibe_input": DataVibe}),  # type: ignore
        output_spec=TypeDictVibe({"processed_data": DataVibe}),
        parameters={},
        entrypoint=EntryPointDict(file="fake.py", callback_builder="fake_callback"),
        root_folder="/tmp",
        description=TaskDescription(),
    )


def test_resolver_empty_dependency(operation_spec: OperationSpec):
    resolver = OperationDependencyResolver()
    empty_dependency = resolver.resolve(operation_spec)

    assert len(empty_dependency) == 0


def test_resolver_valid_dependency(operation_spec: OperationSpec):
    operation_spec.parameters = {"param": 1, "another_param": "test"}
    operation_spec.dependencies = {"parameters": ["param", "another_param"]}

    resolver = OperationDependencyResolver()
    dependencies = resolver.resolve(operation_spec)
    target_dependencoes = {"parameters": operation_spec.parameters}

    assert target_dependencoes == dependencies


def test_resolver_valid_partial_dependency(operation_spec: OperationSpec):
    operation_spec.parameters = {"param": 1, "another_param": "test"}
    operation_spec.dependencies = {"parameters": ["another_param"]}

    resolver = OperationDependencyResolver()
    dependencies = resolver.resolve(operation_spec)
    target_dependencies = {"parameters": {"another_param": "test"}}

    assert target_dependencies == dependencies


def test_resolver_invalid_dependency(operation_spec: OperationSpec):
    operation_spec.parameters = {"param": 1, "another_param": "test"}
    operation_spec.dependencies = {"parameters": ["unexisting_param"]}

    resolver = OperationDependencyResolver()
    with pytest.raises(ValueError):
        resolver.resolve(operation_spec)
