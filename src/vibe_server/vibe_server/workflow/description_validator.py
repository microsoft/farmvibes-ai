# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Dict, List, Tuple, Union

from .parameter import ParameterResolver
from .spec_parser import WorkflowSpec


def unpack_description(description: Union[str, Tuple[str], None]) -> str:
    if isinstance(description, tuple):
        return description[0]
    else:
        return "" if description is None else description


class WorkflowDescriptionValidator:
    @classmethod
    def _validate_node_against_description(
        cls,
        node_name: str,
        node_type: str,
        description: Dict[str, str],
        workflow_name: str,
    ):
        if node_name not in description or not description[node_name]:
            raise ValueError(
                f"{node_type} {node_name} in workflow {workflow_name} is missing a description."
            )

    @classmethod
    def _validate_description_against_nodes(
        cls, desc_nodes: List[str], node_type: str, nodes: Dict[str, Any], workflow_name: str
    ):
        for name in desc_nodes:
            if name not in nodes:
                raise ValueError(
                    f"{name} in the workflow description does not match "
                    f"any {node_type} in workflow {workflow_name}"
                )

    @classmethod
    def _validate_sources(cls, spec: WorkflowSpec):
        for source_name in spec.sources.keys():
            cls._validate_node_against_description(
                source_name, "Source", spec.description.inputs, spec.name
            )

    @classmethod
    def _validate_sinks(cls, spec: WorkflowSpec):
        for sink_name in spec.sinks.keys():
            cls._validate_node_against_description(
                sink_name, "Sink", spec.description.outputs, spec.name
            )

    @classmethod
    def _validate_parameters(cls, workflow_spec: WorkflowSpec):
        param_resolver = ParameterResolver(workflow_spec.workflows_dir, workflow_spec.ops_dir)
        parameters = param_resolver.resolve(workflow_spec)
        param_descriptions = {k: unpack_description(v.description) for k, v in parameters.items()}

        for param_name in workflow_spec.parameters.keys():
            cls._validate_node_against_description(
                param_name, "Parameter", param_descriptions, workflow_spec.name
            )

    @classmethod
    def _validate_tasks(cls, workflow_spec: WorkflowSpec):
        for task_name in workflow_spec.tasks.keys():
            cls._validate_node_against_description(
                task_name, "Task", workflow_spec.description.task_descriptions, workflow_spec.name
            )

    @classmethod
    def _validate_description(cls, spec: WorkflowSpec):
        desc = spec.description
        if not desc.short_description:
            raise ValueError(f"Short description is missing in workflow {spec.name}.")

        # Make sure every node in the description matches to a source/sink/parameter
        for desc_nodes, node_type, node in [
            (desc.inputs, "sources", spec.sources),
            (desc.outputs, "sinks", spec.sinks),
            (desc.parameters, "parameters", spec.parameters),
        ]:
            cls._validate_description_against_nodes(desc_nodes.keys(), node_type, node, spec.name)

    @classmethod
    def validate(cls, workflow_spec: WorkflowSpec):
        cls._validate_sources(workflow_spec)
        cls._validate_sinks(workflow_spec)
        cls._validate_parameters(workflow_spec)
        cls._validate_tasks(workflow_spec)
        cls._validate_description(workflow_spec)
