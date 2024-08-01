# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import os
import re
from collections import defaultdict
from copy import deepcopy
from enum import IntEnum
from re import Pattern
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Set, Tuple, Type, TypeVar, cast

from vibe_common.constants import DEFAULT_OPS_DIR
from vibe_common.schemas import EntryPointDict, OperationSpec
from vibe_core.data.core_types import BaseVibe, DataVibeType, TypeDictVibe, UnresolvedDataVibe
from vibe_core.data.utils import (
    get_base_type,
    get_most_specific_type,
    is_container_type,
    is_vibe_list,
)
from vibe_core.datamodel import TaskDescription

from . import get_workflow_dir
from .graph import Edge, Graph
from .spec_parser import (
    SpecNodeType,
    WorkflowParser,
    WorkflowSpec,
    WorkflowSpecEdge,
    WorkflowSpecNode,
    get_parameter_reference,
    parse_edge_string,
)
from .spec_validator import WorkflowSpecValidator

ORIGIN = 0
DESTINATION = 1
LABEL = 2
T = TypeVar("T", bound=BaseVibe)


class InputFanOut(OperationSpec):
    input_port: str = "input"
    output_port: str = "output"

    def __init__(self, name: str, data_type: DataVibeType):
        if not is_container_type(data_type):
            data_type = List[data_type]  # type: ignore
        inputs_spec = TypeDictVibe({self.input_port: data_type})
        output_spec = TypeDictVibe({self.output_port: data_type})
        ed: EntryPointDict = {"file": "", "callback_builder": ""}
        td = TaskDescription()
        super().__init__(name, "", inputs_spec, output_spec, ed, td, {}, {}, {})


class GraphNodeType(NamedTuple):
    name: str
    spec: OperationSpec


class EdgeType(IntEnum):
    single = 0
    parallel = 1
    scatter = 2
    gather = 3


class EdgeLabel(NamedTuple):
    srcport: str
    dstport: str
    type: EdgeType

    def __hash__(self):
        return hash(self.srcport) * hash(self.dstport)


class WorkflowEdge(Edge[GraphNodeType, EdgeLabel]):
    def __str__(self):
        src, dst, label = self
        return f"{src.name}.{label.srcport} -> {dst.name}.{label.dstport} ({label.type.name})"


class Workflow(Graph[GraphNodeType, EdgeLabel]):
    param_pattern: "Pattern[str]" = re.compile(r"@from\((.*)\)")
    logger: logging.Logger
    workflow_spec: WorkflowSpec
    index: Dict[str, GraphNodeType]
    _sinks: Dict[GraphNodeType, List[str]]
    _sources: Dict[GraphNodeType, List[str]]

    def __init__(self, workflow_spec: WorkflowSpec, resolve: bool = True):
        """Instantiate workflow from a workflow specification.
        Given a workflow specification, instantiate all tasks, recursively instantiating workflows,
        and connect all nodes.
        When `resolve = False`, do not resolve types and edge labels.
        This is necessary when instantiating inner workflows in order to resolve everything
        when the whole graph is in place.
        """
        super().__init__()

        self.logger = logging.getLogger(f"{__name__}.Workflow")
        self.workflow_spec = workflow_spec

        self._build_index()

        self.source_mappings = {k: [i for i in v] for k, v in self.workflow_spec.sources.items()}
        self._sources = defaultdict(list)
        for sources in self.source_mappings.values():
            for source in sources:
                name, port = parse_edge_string(source, maxsplit=-1)
                self._sources[self.index[name]].append(port)

        self.sink_mappings = {k: v for k, v in self.workflow_spec.sinks.items()}
        self._sinks = defaultdict(list)
        for sink in self.sink_mappings.values():
            name, port = parse_edge_string(sink, maxsplit=-1)
            self._sinks[self.index[name]].append(port)

        if resolve:
            self.resolve_types()
            self.validate()

            fanout, fanin = self._find_fan_out_fan_in_edges()
            self._update_edges(fanout, fanin)

    def _ensure_same_container(
        self, input_type: DataVibeType, ref_type: DataVibeType
    ) -> DataVibeType:
        """
        Ensure the input type has (doesn't have) a container if the reference
        type has (does not have) one
        """
        base_type = get_base_type(input_type)
        if is_vibe_list(ref_type):
            return cast(Type[List[BaseVibe]], List[base_type])
        return base_type

    def _resolve_types_for_node(self, node: GraphNodeType):
        """
        Resolve types for all output ports in node
        """
        for port_name in node.spec.output_spec:
            self._resolve_port_type(node, port_name)

    def _resolve_port_type(self, node: GraphNodeType, port_name: str):
        """
        Resolve port type and update the op spec, if necessary.
        This method assumes that the referred port already has a resolved type
        This is the case for our current implementation because we traverse the
        graph in topological order
        """
        port_type = node.spec.output_spec[port_name]
        if not isinstance(port_type, UnresolvedDataVibe):
            # Nothing to resolve
            return

        origin_port = port_type.__name__
        origin_str = f"{node.name}.{origin_port}"
        port_str = f"{node.name}.{port_name}"
        try:
            origin_type = node.spec.inputs_spec[origin_port]
        except KeyError:
            raise ValueError(
                f"Could not infer type of '{port_str}': "
                f"'{origin_port}' is not an input port for '{node.name}'"
            )
        if origin_port in self.sources.get(node, []):
            # There is no one to get the type from because we refer to a source port.
            # We get it from the input port for now, could try something smarter
            self.logger.debug(
                f"Inferring type of {port_str} directly from referenced "
                f"input port {origin_str} because it is a source port"
            )
            node.spec.output_spec[port_name] = origin_type
            return

        # Let's get the type from what connects to the origin port
        source, _, label = self.edge_to(node, origin_port)
        source_port = label.srcport
        source_type = source.spec.output_spec[source_port]

        if isinstance(source_type, UnresolvedDataVibe):
            raise RuntimeError(
                f"Unresolved type on previous level port {source.name}.{source_port}"
            )

        node.spec.output_spec[port_name] = self._ensure_same_container(source_type, origin_type)

    def resolve_types(self):
        for nodes in self.topological_sort():
            for node in nodes:
                self._resolve_types_for_node(node)

    def validate(self) -> bool:
        if self.has_cycle():
            try:
                self.topological_sort()
            except ValueError as e:
                raise ValueError(
                    f"Workflows should be Directed Acyclic Graphs, "
                    f"but workflow {self.workflow_spec.name} has a cycle"
                ) from e
        self._validate_edges_io()
        self._validate_all_inputs_connected()
        self._validate_sinks_exist()
        # We verify compatibility of ports associated to a source when building the inputs spec
        # Calling it here acts as validation of the workflow sources
        self.inputs_spec
        return True

    @property
    def ops_dir(self) -> str:
        return self.workflow_spec.ops_dir

    @property
    def workflow_dir(self) -> str:
        return self.workflow_spec.workflows_dir

    def _get_type_for(self, port_str: str) -> DataVibeType:
        name, port = parse_edge_string(port_str, maxsplit=-1)
        op = self.index[name].spec
        try:
            return op.inputs_spec[port]
        except KeyError:
            return op.output_spec[port]

    def _remove_label_from_edge(
        self, edges: Iterable[Edge[GraphNodeType, EdgeLabel]]
    ) -> Set[Tuple[GraphNodeType, GraphNodeType]]:
        return {e[:-1] for e in edges}

    def _find_fan_out_fan_in_edges(self) -> Tuple[Set[Edge[GraphNodeType, EdgeLabel]], ...]:
        fanout = set()
        fanin = set()
        for edge in self.edges:
            source, destination, label = edge
            srctype = source.spec.output_spec[label.srcport]
            dsttype = destination.spec.inputs_spec[label.dstport]
            if isinstance(srctype, UnresolvedDataVibe):
                raise RuntimeError(
                    f"Unresolved type found on edge {edge}, when finding fan-out/in edges"
                )
            if is_vibe_list(srctype) == is_vibe_list(dsttype):
                continue
            if is_vibe_list(srctype) and not is_vibe_list(dsttype):
                fanout.add(edge)
            elif is_vibe_list(dsttype) and not is_vibe_list(srctype):
                fanin.add(edge)
            else:
                raise RuntimeError(
                    f"srctype {srctype} and dsttype {dsttype} are different "
                    f"but are not of the expected types List -> DataVibe "
                    "or DataVibe -> List"
                )
        return fanout, fanin

    def _update_edges(
        self,
        fanout: Set[Edge[GraphNodeType, EdgeLabel]],
        fanin: Set[Edge[GraphNodeType, EdgeLabel]],
    ):
        op_parallelism = {}
        for edge in fanin:
            self.relabel(edge, EdgeLabel(*edge[LABEL][:-1], EdgeType.gather))
        for edge in fanout:
            self.relabel(edge, EdgeLabel(*edge[LABEL][:-1], EdgeType.scatter))

        for root in self.sources:
            self.propagate_labels(root, 0, op_parallelism)
        for task, v in op_parallelism.items():
            if v < 0:
                raise ValueError(f"Fan-in without parallelism at input of {task.name}")
            if v > 1:
                # This should never happen because we break during propagation
                raise RuntimeError(f"Nested fan-out at input of {task.name}")

    def propagate_labels(
        self, root: GraphNodeType, parallelism_level: int, op_parallelism: Dict[GraphNodeType, int]
    ):
        """Propagate parallelism labels across the graph.

        We update labels according to the parallelism level of previous edges along a path
        (single -> parallel if parallelism_level > 0).

        Our parallelization strategy involves parallelizing ops if *any* of the incoming edges is
        parallel. If there are both parallel and singular edges in the same op, the parallel edges
        distribute items into several instances of the op, while all the data flowing into singular
        edges is replicated as is to all op instances.
        Due to this strategy, we keep track of the maximum parallelism level of all input ports
        in an op, and propagate that into the next level. This means that in some paths the
        algorithm might temporarily assign wrong parallelism levels to edges (even < 0), but they
        will be overwritten to the correct level after the most parallel path is traversed.
        """
        for source, neighbor, label in self.edges_from(root):
            edge = WorkflowEdge((source, neighbor, label))
            label_type = label.type
            neighbor_parallelism_level = parallelism_level
            if label_type == EdgeType.parallel:
                return
            elif label_type == EdgeType.single:
                if neighbor_parallelism_level > 0:
                    label_type = EdgeType.parallel
            elif label_type == EdgeType.scatter:
                if neighbor_parallelism_level > 0:
                    raise ValueError(f"Nested fan-out found at edge {edge} is unsupported")
                neighbor_parallelism_level += 1
            elif label_type == EdgeType.gather:
                # If we are not parallel, gather will just make a list of a single element
                neighbor_parallelism_level = max(0, neighbor_parallelism_level - 1)
            else:
                raise RuntimeError(f"Found unknown label type in edge {edge}")
            if neighbor in op_parallelism:
                neighbor_parallelism_level = max(
                    neighbor_parallelism_level, op_parallelism[neighbor]
                )
            op_parallelism[neighbor] = neighbor_parallelism_level
            self.relabel((source, neighbor, label), EdgeLabel(*label[:-1], label_type))
            self.propagate_labels(neighbor, neighbor_parallelism_level, op_parallelism)

    def prefix_node(self, node: GraphNodeType, prefix: str) -> GraphNodeType:
        return GraphNodeType(name=f"{prefix}.{node.name}", spec=node.spec)

    def merge_inner_workflow(self, inner_workflow: "Workflow", prefix: str):
        inner_index = {
            f"{prefix}.{k}": self.prefix_node(v, prefix) for k, v in inner_workflow.index.items()
        }
        # Add nodes to the graph
        for v in inner_index.values():
            self.add_node(v)
        # Update our index
        self.index.update(inner_index)
        # Add edges
        for edge in inner_workflow.edges:
            origin, destination, label = edge
            self.add_edge(
                inner_index[f"{prefix}.{origin.name}"],
                inner_index[f"{prefix}.{destination.name}"],
                label,
            )

    def _load_inner_workflow(self, workflow: WorkflowSpec, taskname: str) -> None:
        wf = Workflow(workflow, resolve=False)
        spec = wf.workflow_spec
        self.workflow_spec.edges = list(
            self._update_workflow_spec_edges(self.workflow_spec.edges, spec, taskname)
        )
        self.workflow_spec.sources = dict(
            self._update_workflow_spec_sources(self.workflow_spec.sources, spec, taskname)
        )
        self.workflow_spec.sinks = dict(
            self._update_workflow_spec_sinks(self.workflow_spec.sinks, spec, taskname)
        )
        self.merge_inner_workflow(wf, taskname)

    def _add_workflow_edge_to_graph(self, origin: str, destination: str) -> None:
        origin, srcport = parse_edge_string(origin, -1)
        destination, dstport = parse_edge_string(destination, -1)
        try:
            if srcport not in self.index[origin].spec.output_spec:
                raise ValueError(f"Port {srcport} could not be found as output of op {origin}")
            if dstport not in self.index[destination].spec.inputs_spec:
                raise ValueError(f"Port {dstport} could not be found as input of op {destination}")
            self.add_edge(
                self.index[origin],
                self.index[destination],
                EdgeLabel(srcport, dstport, EdgeType.single),
            )
        except KeyError as e:
            raise ValueError(
                f"Tried to connect port {srcport} from op {origin} to "
                f"port {dstport} of op {destination}, but {str(e)} does "
                "not exist in the workflow graph."
            )

    def _resolve_parameters(self, task: SpecNodeType):
        wf_params = self.workflow_spec.parameters

        def resolve(parameters: Dict[str, Any], default: Dict[str, Any]):
            new_params = deepcopy(parameters)
            for k, v in parameters.items():
                if isinstance(v, dict):
                    new_params[k] = resolve(parameters[k], default[k])
                ref_name = get_parameter_reference(v, task.name)
                if ref_name is not None:
                    if wf_params is None or ref_name not in wf_params:
                        raise ValueError(
                            f"Could not find parameter '{ref_name}' in workflow '{self.name}'"
                            f" to substitute in task '{task.name}'"
                        )
                    override = wf_params[ref_name]
                    # Keep default parameter if override is not defined
                    new_params[k] = default[k] if override is None else override
            return new_params

        task.parameters = resolve(task.parameters, task.default_parameters)

    def _build_index(self) -> Dict[str, GraphNodeType]:
        self.index: Dict[str, GraphNodeType] = {}

        for k, t in self.workflow_spec.tasks.items():
            task = t.load(self.ops_dir, self.workflow_dir)
            self._resolve_parameters(task)
            if isinstance(task, WorkflowSpec):
                self._load_inner_workflow(task, k)
            else:
                self.index[k] = GraphNodeType(k, task)
                self.add_node(self.index[k])
        for edge in self.workflow_spec.edges:
            for destination in edge.destination:
                self._add_workflow_edge_to_graph(edge.origin, destination)

        return self.index

    def _update_workflow_spec_sources(
        self,
        sources: Dict[str, List[str]],
        included_workflow_spec: WorkflowSpec,
        prefix: str,
    ) -> Iterable[Tuple[str, List[str]]]:
        for sourcename, targets in sources.items():
            tmp = []
            for target in targets:
                target_task, target_source_name = parse_edge_string(target, -1)
                if target_task != prefix:
                    tmp.append(target)
                else:
                    if target_source_name not in included_workflow_spec.sources:
                        raise ValueError(
                            f"Could not find source '{target_source_name}' "
                            f"in inner workflow '{prefix}'"
                        )
                    tmp.extend(
                        [
                            f"{prefix}.{t}"
                            for t in included_workflow_spec.sources[target_source_name]
                        ]
                    )
            yield sourcename, tmp

    def _update_workflow_spec_sinks(
        self,
        sinks: Dict[str, str],
        included_workflow_spec: WorkflowSpec,
        prefix: str,
    ) -> Iterable[Tuple[str, str]]:
        for name, real_sink in sinks.items():
            sink_task, sink_name = parse_edge_string(real_sink, -1)
            if sink_task != prefix:
                yield name, real_sink
            else:
                if sink_name not in included_workflow_spec.sinks:
                    raise ValueError(
                        f"Could not find sink '{sink_name}' in inner workflow '{prefix}'"
                    )
                yield name, f"{prefix}.{included_workflow_spec.sinks[sink_name]}"

    def _update_workflow_spec_edges(
        self, edges: List[WorkflowSpecEdge], included_workflow_spec: WorkflowSpec, prefix: str
    ) -> Iterable[WorkflowSpecEdge]:
        for edge in edges:
            tmp = self._update_edge_destinations(edge, included_workflow_spec, prefix)
            yield self._update_edge_origin(tmp, included_workflow_spec, prefix)

    def _update_edge_destinations(
        self, edge: WorkflowSpecEdge, included_workflow_spec: WorkflowSpec, prefix: str
    ) -> WorkflowSpecEdge:
        new_edge = WorkflowSpecEdge(edge.origin, [])
        for destination in edge.destination:
            matched = False
            for source, targets in included_workflow_spec.sources.items():
                sourcename = f"{prefix}.{source}"
                if destination == sourcename:
                    new_edge.destination.extend(
                        [f"{prefix}.{target}" for target in targets],
                    )
                    # Mask the match
                    matched = True
                    # If we match one source, we won't match others, so we're done
                    break
            if not matched:
                # We don't have any matches, let's put it back in the list
                new_edge.destination.append(destination)
        return new_edge

    def _update_edge_origin(
        self, edge: WorkflowSpecEdge, included_workflow_spec: WorkflowSpec, prefix: str
    ) -> WorkflowSpecEdge:
        for spec_name, real_name in included_workflow_spec.sinks.items():
            if f"{prefix}.{spec_name}" == edge.origin:
                edge.origin = f"{prefix}.{real_name}"
                # We updated the edge, our work is done
                return edge
        return edge

    def _validate_all_inputs_connected(self):
        inputs = {
            f"{name}.{port}": False
            for name, node in self.index.items()
            for port in node.spec.inputs_spec
        }

        for node, ports in self.sources.items():
            for port in ports:
                key = f"{node.name}.{port}"
                if key not in inputs:
                    raise ValueError(f"'{key}' not in inputs dictionary")
                inputs[key] = True

        for _, destination, label in self.edges:
            key = f"{destination.name}.{label.dstport}"
            if key not in inputs:
                raise ValueError(f"'{key}' not in inputs dictionary")
            inputs[key] = True

        missing: List[str] = []
        for key, value in inputs.items():
            if not value:
                missing.append(f"'{key}'")

        if missing:
            s = "s" if len(missing) > 1 else ""
            raise ValueError(
                f"Operation{s} port{s} {','.join(missing)} missing inputs. "
                "All tasks in a workflow must have all their inputs filled"
            )
        return True

    def _validate_edges_io(self):
        def check_compatible_io(edge: WorkflowEdge) -> None:
            origin, destination, label = edge
            origin_type = get_base_type(origin.spec.output_spec[label.srcport])
            destination_type = get_base_type(destination.spec.inputs_spec[label.dstport])
            if not issubclass(origin_type, destination_type):
                raise ValueError(
                    "Incompatible types for edge "
                    f'"{origin.name}.{label.srcport}" ({origin_type.__name__})'
                    f' -> "{destination.name}.{label.dstport}" ({destination_type.__name__})'
                )

        for edge in self.edges:
            check_compatible_io(edge)

    def _validate_sinks_exist(self):
        for node, ports in self.sinks.items():
            for port in ports:
                if port not in node.spec.output_spec:
                    raise ValueError(f"'{node.name}.{port}' not in op output spec")

    def __getitem__(self, op_name: str) -> OperationSpec:
        for op in self.nodes:
            if op.name == op_name:
                return op.spec
        raise KeyError(f"op {op_name} does not exist")

    @property
    def name(self):
        return self.workflow_spec.name

    @property
    def inputs_spec(self) -> TypeDictVibe:
        spec = {}
        for k, v in self.source_mappings.items():
            try:
                spec[k] = get_most_specific_type([self._get_type_for(i) for i in v])
            except ValueError as e:
                raise ValueError(f"Workflow source '{k}' contains incompatible types. {e}")
        return TypeDictVibe(spec)

    @property
    def output_spec(self):
        return TypeDictVibe({k: self._get_type_for(v) for k, v in self.sink_mappings.items()})

    @property
    def sources(self) -> Dict[GraphNodeType, List[str]]:
        return {k: v for k, v in self._sources.items()}

    @property
    def sinks(self) -> Dict[GraphNodeType, List[str]]:
        return {k: v for k, v in self._sinks.items()}

    @property
    def edges(self) -> List[WorkflowEdge]:
        return [WorkflowEdge(e) for e in super().edges]

    def edges_from(self, node: GraphNodeType) -> List[WorkflowEdge]:
        return [WorkflowEdge(e) for e in super().edges_from(node)]

    def edge_to(self, node: GraphNodeType, port_name: str):
        edges = [e for e in self.edges if e[LABEL].dstport == port_name and e[DESTINATION] is node]
        port_str = f"'{node.name}.{port_name}'"
        if not edges:
            raise ValueError(f"{port_str} is not a destination of any port")
        if len(edges) > 1:
            # Something went very wrong if we are here
            raise RuntimeError(f"Found multiple edges with '{port_str}' as destination")
        return edges[0]

    def get_node(self, op_name: str) -> WorkflowSpecNode:
        return self.workflow_spec.tasks[op_name]

    def get_op_dir(self, op_name: str) -> Optional[str]:
        return self.workflow_spec.tasks[op_name].op_dir

    def get_op_parameter(self, op_name: str) -> Optional[Dict[str, Any]]:
        return self.workflow_spec.tasks[op_name].parameters

    @classmethod
    def build(
        cls,
        workflow_path: str,
        ops_base_dir: str = DEFAULT_OPS_DIR,
        workflow_base_dir: str = get_workflow_dir(),
        parameters_override: Optional[Dict[str, Any]] = None,
    ) -> "Workflow":
        spec = WorkflowParser.parse(
            workflow_path,
            ops_base_dir,
            workflow_base_dir,
            parameters_override,
        )
        WorkflowSpecValidator.validate(spec)
        return Workflow(spec)


def load_workflow_by_name(
    name: str,
    ops_dir: str = DEFAULT_OPS_DIR,
    workflow_dir: str = get_workflow_dir(),
) -> Workflow:
    """Loads a workflow in the format returned by `list_workflows()`"""

    return Workflow.build(
        os.path.join(workflow_dir, f"{name}.yaml"),
        ops_base_dir=ops_dir,
        workflow_base_dir=workflow_dir,
    )
