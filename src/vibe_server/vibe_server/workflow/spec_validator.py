# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import warnings
from typing import List

from vibe_common.schemas import OperationSpec

from .parameter import ParameterResolver
from .spec_parser import (
    SpecNodeType,
    WorkflowSpec,
    flat_params,
    get_parameter_reference,
    parse_edge_string,
)


class WorkflowSpecValidator:
    @classmethod
    def _validate_node_exists(cls, spec: WorkflowSpec, nodename: str, type: str) -> bool:
        if nodename not in spec.tasks:
            raise ValueError(
                f"Workflow {spec.name} specifies node {nodename} as {type}, but it doesn't exist"
            )
        return True

    @classmethod
    def _validate_sources(cls, spec: WorkflowSpec) -> bool:
        mapping_error = "Sources field must be a mapping between strings and lists of strings"
        if not isinstance(spec.sources, dict):
            raise ValueError(mapping_error)
        else:
            for k, v in spec.sources.items():
                if not (isinstance(k, str) and isinstance(v, list)):
                    raise ValueError(mapping_error)

        if len(spec.sources) == 0:
            raise ValueError(f"There must be at least one source in workflow spec {spec.name}.")

        for source_name, source_ports in spec.sources.items():
            if len(source_ports) == 0:
                raise ValueError(
                    f"Source {source_name} must be associated with at least "
                    f"one task input in workflow spec {spec.name}."
                )

        return cls._validate_node_list(
            spec, [e for v in spec.sources.values() for e in v], "source"
        )

    @classmethod
    def _validate_sinks(cls, spec: WorkflowSpec) -> bool:
        mapping_error = "Sinks field must be a mapping of strings"
        if not isinstance(spec.sinks, dict):
            raise ValueError(mapping_error)
        else:
            for k, v in spec.sinks.items():
                if not (isinstance(k, str) and isinstance(v, str)):
                    raise ValueError(mapping_error)

        if len(spec.sinks) == 0:
            warnings.warn(
                f"Workflow {spec.name} has no sinks. Is it being used for side-effects only?"
            )

        return cls._validate_node_list(spec, [v for v in spec.sinks.values()], "sink")

    @classmethod
    def _validate_node_list(cls, spec: WorkflowSpec, ref: List[str], type: str) -> bool:
        for thing in ref:
            nodename, _ = parse_edge_string(thing)
            cls._validate_node_exists(spec, nodename, type)
        return True

    @classmethod
    def _port_exists(cls, port: str, node: SpecNodeType) -> bool:
        if isinstance(node, OperationSpec):
            return port in node.inputs_spec or port in node.output_spec
        return port in node.sources or port in node.sinks

    @classmethod
    def _validate_workflow_without_edges(cls, workflow_spec: WorkflowSpec) -> bool:
        sink_nodes = set([parse_edge_string(s)[0] for s in workflow_spec.sinks.values()])
        source_nodes = set(
            [parse_edge_string(ss)[0] for s in workflow_spec.sources.values() for ss in s]
        )
        task_nodes = workflow_spec.tasks
        if not len(task_nodes) == len(sink_nodes) == len(source_nodes):
            raise ValueError(
                "The number of sink and source nodes should equal the number of tasks "
                "when defining a workflow without edges."
            )
        # "Single"-operation workflows aren't required to have edges
        workflow_spec.edges = []
        return True

    @classmethod
    def _validate_edges(cls, workflow_spec: WorkflowSpec) -> bool:
        if not workflow_spec.edges:
            cls._validate_workflow_without_edges(workflow_spec)
        if not isinstance(workflow_spec.edges, list):
            raise TypeError(f"Edges of workflow {workflow_spec.name} are not in a list.")
        source_ports = [port for source in workflow_spec.sources.values() for port in source]
        for edge in workflow_spec.edges:
            if not isinstance(edge.destination, list):
                raise TypeError(f"Destination of edge {edge} is not a list")
            for source in source_ports:
                if source in edge.destination:
                    raise ValueError(
                        f"Source {source} is also a destination of edge "
                        f"{edge.origin} -> {source}"
                    )
            cls._validate_node_list(workflow_spec, [edge.origin], "edge origin")
            cls._validate_node_list(workflow_spec, edge.destination, "edge destination")
        return True

    @classmethod
    def _validate_parameter_references(cls, workflow_spec: WorkflowSpec):
        """
        Validate that all defined workflow parameters are used in tasks and that all parameter
        references exist
        """

        param_references = {
            get_parameter_reference(v, task_name)
            for task_name, task in workflow_spec.tasks.items()
            for v in flat_params(task.parameters)
        }
        param_references.discard(None)
        bad_params = [param for param in workflow_spec.parameters if param not in param_references]
        bad_references = {ref for ref in param_references if ref not in workflow_spec.parameters}
        if not (bad_params or bad_references):
            return
        error_msg = []
        for msg, bad_stuff in zip(
            (
                "Workflow parameter{s} {bad_stuff_str} {is_are} not mapped to any task parameters",
                "Task parameters reference undefined workflow parameter{s} {bad_stuff_str}",
            ),
            (bad_params, bad_references),
        ):
            if bad_stuff:
                bad_stuff_str = ", ".join([f"'{i}'" for i in bad_stuff])
                s = "s" if len(bad_stuff) > 1 else ""
                is_are = "are" if len(bad_stuff) > 1 else "is"
                error_msg.append(msg.format(bad_stuff_str=bad_stuff_str, s=s, is_are=is_are))
        raise ValueError(". ".join(error_msg))

    @classmethod
    def _validate_parameter_defaults(cls, workflow_spec: WorkflowSpec):
        resolver = ParameterResolver(workflow_spec.workflows_dir, workflow_spec.ops_dir)
        params = resolver.resolve(workflow_spec)
        bad_params = [k for k, v in params.items() if isinstance(v.default, tuple)]
        if bad_params:
            param_names = ", ".join([f"'{p}'" for p in bad_params])
            s = "s" if len(bad_params) > 1 else ""
            s_ = "" if len(bad_params) > 1 else "s"
            raise ValueError(
                f"Workflow parameter{s} {param_names} map{s_} to task parameters with different "
                "default values. Please define a default value in the workflow."
            )

    @classmethod
    def _validate_parameters(cls, workflow_spec: WorkflowSpec):
        cls._validate_parameter_references(workflow_spec)
        cls._validate_parameter_defaults(workflow_spec)

    @classmethod
    def validate(cls, workflow_spec: WorkflowSpec) -> WorkflowSpec:
        cls._validate_sources(workflow_spec)
        cls._validate_sinks(workflow_spec)
        cls._validate_edges(workflow_spec)
        cls._validate_parameters(workflow_spec)

        for task in workflow_spec.tasks.values():
            spec = task.load(workflow_spec.ops_dir, workflow_spec.workflows_dir)
            if isinstance(spec, WorkflowSpec):
                if spec.name == workflow_spec.name:
                    raise ValueError(
                        f"Recursive definition of workflow {workflow_spec.name} is not supported."
                    )
                cls.validate(spec)

        return workflow_spec
