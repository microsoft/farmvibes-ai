# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import re
from copy import deepcopy
from dataclasses import dataclass
from enum import auto
from re import Pattern
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, cast

import yaml
from fastapi_utils.enums import StrEnum

from vibe_common.constants import DEFAULT_OPS_DIR
from vibe_common.schemas import OperationParser, OperationSpec, update_parameters
from vibe_core.datamodel import TaskDescription
from vibe_core.utils import (
    MermaidVerticesMap,
    build_mermaid_edge,
    draw_mermaid_diagram,
    rename_keys,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEV_WORKFLOW_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "workflows"))
RUN_WORKFLOW_DIR = os.path.join("/", "app", "workflows")

PARAM_PATTERN: "Pattern[str]" = re.compile(r"@from\((.*)\)")


def get_workflow_dir() -> str:
    if os.path.exists(DEV_WORKFLOW_DIR):
        return DEV_WORKFLOW_DIR
    else:
        return RUN_WORKFLOW_DIR


def get_parameter_reference(param: Any, task_name: str) -> Optional[str]:
    if isinstance(param, str) and (match := re.match(PARAM_PATTERN, param)):
        param_msg = f"task '{task_name}', parameter '{param}'"
        if len(g := match.groups()) > 1:
            raise ValueError(f"Failed to parse parameter reference '{param}' in {param_msg}")
        ref_name = g[0]
        if not ref_name:
            raise ValueError(f"Found empty parameter reference in {param_msg}")
        return ref_name
    return None


def split_task_name_port(edge: str) -> Tuple[str, str]:
    spllited_edge = edge.split(".")
    return tuple(spllited_edge[0:1] + spllited_edge[-1:])


SpecNodeType = Union[OperationSpec, "WorkflowSpec"]


class TaskType(StrEnum):
    op = auto()
    workflow = auto()


@dataclass
class WorkflowSpecEdge:
    origin: str
    destination: List[str]


@dataclass
class WorkflowSpecNode:
    task: str
    type: TaskType
    parameters: Dict[str, Any]
    op_dir: Optional[str]  # only exists when this is an op
    parent: str  # the workflow that gave rise to this

    def load(
        self, ops_base_dir: str = DEFAULT_OPS_DIR, workflow_dir: str = get_workflow_dir()
    ) -> SpecNodeType:
        if self.type == TaskType.op:
            return self._load_op(ops_base_dir)
        else:
            return self._load_workflow(ops_base_dir, workflow_dir)

    def _load_op(self, ops_base_dir: str) -> OperationSpec:
        assert isinstance(self.op_dir, str)
        return OperationParser.parse(
            os.path.abspath(os.path.join(ops_base_dir, self.op_dir, f"{self.task}.yaml")),
            self.parameters,
        )

    def _load_workflow(self, ops_base_dir: str, workflow_dir: str) -> "WorkflowSpec":
        return WorkflowParser.parse(
            os.path.abspath(os.path.join(workflow_dir, f"{self.task}.yaml")),
            ops_base_dir,
            workflow_dir,
            self.parameters,
        )

    def to_dict(self) -> Dict[str, Any]:
        ret = {
            ("op" if self.type == TaskType.op else "workflow"): self.task,
            "parameters": self.parameters,
        }
        if self.op_dir is not None:
            ret["op_dir"] = self.op_dir
        return ret


@dataclass
class WorkflowSpec:
    name: str
    sources: Dict[str, List[str]]
    sinks: Dict[str, str]
    tasks: Dict[str, WorkflowSpecNode]
    edges: List[WorkflowSpecEdge]
    parameters: Dict[str, Any]
    default_parameters: Dict[str, Any]
    description: TaskDescription
    ops_dir: str
    workflows_dir: str

    def __post_init__(self):
        for i, e in enumerate((e for e in self.edges)):
            if isinstance(e, dict):
                self.edges[i] = WorkflowSpecEdge(**e)
        for k, v in zip(self.tasks.keys(), (v for v in self.tasks.values())):
            if isinstance(v, dict):
                self.tasks[k] = WorkflowSpecNode(**v)
        if isinstance(self.description, dict):
            self.description = TaskDescription(**self.description)
        for task_name, node_spec in self.tasks.items():
            if task_name in self.description.task_descriptions:
                continue
            spec = node_spec.load(self.ops_dir, self.workflows_dir)
            if isinstance(spec.description, dict):
                spec.description = TaskDescription(**spec.description)
            self.description.task_descriptions[task_name] = spec.description.short_description

    def _build_vertices_map(self) -> MermaidVerticesMap:
        vertices = MermaidVerticesMap(sources={}, sinks={}, tasks={})
        # Create a dictionary to map sources, sinks, and tasks to vertex ids
        for i, source in enumerate(self.sources.keys()):
            vertices.sources[source] = f"inp{i+1}>{source}]"
        for i, sink in enumerate(self.sinks.keys()):
            vertices.sinks[sink] = f"out{i+1}>{sink}]"
        for i, task in enumerate(self.tasks.keys()):
            vertices.tasks[task] = f"tsk{i+1}" + "{{" + task + "}}"
        return vertices

    def to_mermaid(self) -> str:
        vertices_map: MermaidVerticesMap = self._build_vertices_map()

        # Create edges between tasks
        edges = [
            build_mermaid_edge(
                split_task_name_port(edge.origin),
                split_task_name_port(destination),
                vertices_map.tasks,
                vertices_map.tasks,
            )
            for edge in self.edges
            for destination in edge.destination
        ]

        # Create edges between sources and tasks
        edges += [
            build_mermaid_edge(
                (source_name, ""),
                split_task_name_port(source_port),
                vertices_map.sources,
                vertices_map.tasks,
            )
            for source_name, source_ports in self.sources.items()
            for source_port in source_ports
        ]

        # Create edges between tasks and sinks
        edges += [
            build_mermaid_edge(
                split_task_name_port(sink_port),
                (sink_name, ""),
                vertices_map.tasks,
                vertices_map.sinks,
            )
            for sink_name, sink_port in self.sinks.items()
        ]

        return draw_mermaid_diagram(vertices_map, edges)


class WorkflowParser:
    required_fields: List[str] = "name sources sinks tasks".split()
    optional_fields: List[str] = "parameters default_parameters edges description".split()
    op_spec_fields: List[str] = "op parameters op_dir".split()
    wf_spec_fields: List[str] = "workflow parameters".split()

    @classmethod
    def _load_workflow(cls, yamlpath: str) -> Dict[str, Any]:
        with open(yamlpath) as fp:
            data = yaml.safe_load(fp)

        return data

    @classmethod
    def _parse_nodespec(
        cls, nodespec: Dict[str, Union[str, Dict[str, Any]]], workflow_name: str, task_name: str
    ) -> WorkflowSpecNode:
        if "workflow" in nodespec:
            type = TaskType.workflow
            possible_fields = cls.wf_spec_fields
        elif "op" in nodespec:
            type = TaskType.op
            possible_fields = cls.op_spec_fields
        else:
            raise ValueError(f"Task specification is missing fields 'op' or 'workflow': {nodespec}")

        task = nodespec[type]
        check_config_fields(nodespec, possible_fields, "Task", task_name)

        # Check field types
        if not isinstance(task, str):
            raise TypeError(f"'{type}' field of task {task_name} is not a string")
        if "parameters" in nodespec and not isinstance(nodespec["parameters"], dict):
            raise TypeError(f"'parameters' field of task {task_name} is not a dictionary")
        if "op_dir" in nodespec and not isinstance(nodespec["op_dir"], str):
            raise TypeError(f"'op_dir' field of task {task_name} is not a dictionary")

        return WorkflowSpecNode(
            task=task,
            type=type,
            parameters=cast(Dict[str, Any], nodespec.get("parameters", {})),
            op_dir=cast(str, nodespec.get("op_dir", task)),
            parent=workflow_name,
        )

    @classmethod
    def _parse_edgespec(cls, edgespec: Dict[str, Union[str, List[str]]]) -> WorkflowSpecEdge:
        return WorkflowSpecEdge(
            origin=cast(str, edgespec["origin"]),
            destination=cast(List[str], edgespec["destination"]),
        )

    @classmethod
    def _workflow_spec_from_yaml_dict(
        cls,
        workflow_dict: Dict[str, Any],
        ops_dir: str,
        workflows_dir: str,
        parameters: Dict[str, Any],
        default_parameters: Dict[str, Any],
    ):
        workflow_name = workflow_dict.get("name", "UNAMED")
        for field in cls.required_fields:
            if field not in workflow_dict:
                raise ValueError(
                    f"Workflow specification '{workflow_name}' is missing required field '{field}'"
                )
        check_config_fields(
            workflow_dict, cls.required_fields + cls.optional_fields, "Workflow", workflow_name
        )
        try:
            edges: Optional[List[Dict[str, Union[str, List[str]]]]] = workflow_dict.get("edges", [])
            if edges is None:
                edges = []
            if not isinstance(edges, list):
                raise TypeError(f"Expected edges to be a list, found {type(edges)}")
            yaml_description: Dict[str, Any] = workflow_dict.get("description", {})
            if yaml_description is None:
                yaml_description = {}
            yaml_description = rename_keys(
                yaml_description, {"sources": "inputs", "sinks": "outputs"}
            )
            description: TaskDescription = TaskDescription(
                **{k: v for k, v in yaml_description.items() if v is not None}
            )
            return WorkflowSpec(
                name=workflow_dict["name"],
                sources=workflow_dict["sources"],
                sinks=workflow_dict["sinks"],
                tasks={
                    k: cls._parse_nodespec(v, workflow_name, k)
                    for k, v in workflow_dict["tasks"].items()
                },
                edges=[cls._parse_edgespec(e) for e in edges],
                parameters=parameters,
                default_parameters=default_parameters,
                description=description,
                ops_dir=ops_dir,
                workflows_dir=workflows_dir,
            )
        except KeyError as e:
            raise ValueError(f"Workflow spec {workflow_dict} is missing field {e}") from e

    @classmethod
    def parse_dict(
        cls,
        workflow_dict: Dict[str, Any],
        ops_dir: str = DEFAULT_OPS_DIR,
        workflows_dir: str = get_workflow_dir(),
        parameters_override: Optional[Dict[str, Any]] = None,
    ) -> "WorkflowSpec":
        params = workflow_dict.get("parameters", {})
        if params is None:
            params = {}
        workflow_dict["default_parameters"] = deepcopy(params)
        if parameters_override is not None:
            params = update_parameters(params, parameters_override)
        workflow_dict["parameters"] = params
        try:
            # workflow_dict is a WorkflowSpec that was serialized to a dict
            return WorkflowSpec(**workflow_dict)
        except TypeError:
            # workflow_dict was loaded from a YAML
            return cls._workflow_spec_from_yaml_dict(
                workflow_dict,
                ops_dir,
                workflows_dir,
                workflow_dict["parameters"],
                workflow_dict["default_parameters"],
            )

    @classmethod
    def parse(
        cls,
        workflow_name: str,
        ops_dir: str = DEFAULT_OPS_DIR,
        workflows_dir: str = get_workflow_dir(),
        parameters_override: Optional[Dict[str, Any]] = None,
    ) -> "WorkflowSpec":
        data = cls._load_workflow(workflow_name)
        return cls.parse_dict(
            data,
            ops_dir,
            workflows_dir,
            parameters_override,
        )


def parse_edge_string(edge_string: str, maxsplit: int = 1) -> Tuple[str, str]:
    return (
        ".".join(edge_string.split(".", maxsplit=maxsplit)[:-1]),
        edge_string.split(".", maxsplit=maxsplit)[-1],
    )


def check_config_fields(
    fields: Iterable[str], accepted_fields: List[str], config_type: str, config_name: str
):
    bad_fields = [field for field in fields if field not in accepted_fields]
    if bad_fields:
        bad_fields_str = ", ".join([f"'{field}'" for field in bad_fields])
        s = "s" if len(bad_fields) > 1 else ""
        raise ValueError(
            f"{config_type} spec '{config_name}' contains unknown field{s} {bad_fields_str}"
        )


def flat_params(params: Dict[str, Any]):
    for param in params.values():
        if isinstance(param, dict):
            yield from flat_params(param)
        else:
            yield param
