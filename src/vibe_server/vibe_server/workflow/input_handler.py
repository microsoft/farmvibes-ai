# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
from typing import (
    Any,
    Dict,
    List,
    Union,
    _type_repr,  # type: ignore
    cast,
)

from vibe_common.input_handlers import gen_stac_item_from_bounds
from vibe_core.data.core_types import DataVibeType, InnerIOType, OpIOType, TypeDictVibe
from vibe_core.data.utils import StacConverter, deserialize_stac, get_base_type, is_container_type
from vibe_core.datamodel import SpatioTemporalJson
from vibe_core.utils import ensure_list

from .workflow import EdgeLabel, EdgeType, GraphNodeType, InputFanOut, Workflow, parse_edge_string

LOGGER = logging.getLogger(__name__)


def add_node(workflow: Workflow, node: GraphNodeType):
    workflow.index[node.name] = node
    workflow.add_node(node)

    def rollback():
        del workflow.adjacency_list[node]
        del workflow.index[node.name]

    return rollback


def source_to_edge(workflow: Workflow, fan_node: GraphNodeType, source: str, destination: str):
    output_port = cast(InputFanOut, fan_node.spec).output_port
    workflow._add_workflow_edge_to_graph(f"{fan_node.name}.{output_port}", destination)
    node_name, port_name = parse_edge_string(destination, maxsplit=-1)
    workflow.sources[workflow.index[node_name]].remove(port_name)
    if not workflow.sources[workflow.index[node_name]]:
        del workflow._sources[workflow.index[node_name]]
        workflow.source_mappings[source].remove(destination)

    def rollback():
        workflow._sources[workflow.index[node_name]].append(port_name)
        workflow.source_mappings[source].append(f"{node_name}.{port_name}")

    return rollback


def add_fan_source(workflow: Workflow, node: GraphNodeType, source: str):
    input_port = cast(InputFanOut, node.spec).input_port
    workflow._sources[workflow.index[node.name]] = [input_port]
    workflow.source_mappings[source].append(f"{node.name}.{input_port}")

    def rollback():
        del workflow._sources[node]
        workflow.source_mappings[source].remove(f"{node.name}.{input_port}")

    return rollback


def recompute_parallelism(workflow: Workflow):
    for edge in workflow.edges:
        new_label = EdgeLabel(*edge[-1][:-1], EdgeType.single)
        workflow.relabel(edge, new_label)
    fanout, fanin = workflow._find_fan_out_fan_in_edges()
    workflow._update_edges(fanout, fanin)


def rollback_parallelism(workflow: Workflow):
    def rollback():
        recompute_parallelism(workflow)

    return rollback


def fan_out_workflow_source(workflow: Workflow, source: str):
    rollback_list = []
    try:
        op_name = f"{source}_fanout"
        fan_node = GraphNodeType(op_name, spec=InputFanOut(op_name, workflow.inputs_spec[source]))
        rollback_list.append(add_node(workflow, fan_node))
        destinations = workflow.source_mappings[source].copy()
        for destination in destinations:
            rollback_list.insert(0, source_to_edge(workflow, fan_node, source, destination))
        rollback_list.insert(0, add_fan_source(workflow, fan_node, source))
        rollback_list.append(rollback_parallelism(workflow))
        recompute_parallelism(workflow)
    except Exception:
        # Something went wrong, let's rollback all changes to the workflow!
        for foo in rollback_list:
            foo()
        raise


def build_args_for_workflow(
    user_input: Union[List[Any], Dict[str, Any], SpatioTemporalJson], wf_inputs: List[str]
) -> OpIOType:
    """
    Get user input and transform it into a dict where the keys match the workflow sources
    """
    # If all the keys match, there is nothing to do
    if isinstance(user_input, dict) and set(wf_inputs) == set(user_input.keys()):
        return user_input
    # Check if there is only one source. If that's the case, assign input to it, otherwise break
    if len(wf_inputs) > 1:
        raise ValueError(
            "User input does not specify workflow sources and workflow has multiple sources: "
            f"{', '.join(wf_inputs)}. A dictionary with matching keys is required."
        )
    # Check if it's a spatiotemporal json (geom + time range)
    # If that's the case we generate a DataVibe with that info
    if isinstance(user_input, SpatioTemporalJson):
        user_input = gen_stac_item_from_bounds(
            user_input.geojson,  # type: ignore
            user_input.start_date,  # type: ignore
            user_input.end_date,  # type: ignore
        )
    return {wf_inputs[0]: user_input}


def validate_workflow_input(user_input: OpIOType, inputs_spec: TypeDictVibe):
    """
    Validate workflow input by making sure user input types match the respective source types
    """
    for source_name, source_type in inputs_spec.items():
        source_input = user_input[source_name]
        validate_vibe_types(source_input, source_type, source_name)


def validate_vibe_types(source_input: InnerIOType, source_type: DataVibeType, source_name: str):
    # If it's a DataVibe, we deserialize and check if the types are compatible
    base_type = get_base_type(source_type)
    try:
        vibe_input = StacConverter().from_stac_item(deserialize_stac(source_input))
    except Exception:
        raise ValueError(
            "Failed to convert inputs to workflow source "
            f"{source_name} of type {_type_repr(source_type)}"
        )
    source_types = set(type(i) for i in ensure_list(vibe_input))
    bad_types = [t for t in source_types if not issubclass(t, base_type)]
    if bad_types:
        raise ValueError(
            f"Workflow source {source_name} expects inputs of type {source_type}, "
            f"found incompatible types: {', '.join(_type_repr(t) for t in bad_types)}"
        )


def patch_workflow_source(source_input: InnerIOType, workflow: Workflow, source_name: str):
    # Check if input is list and type is not list
    # If that's the case, try to patch the workflow with a source fan-out node
    # An element in a list source is fine because we make a one element list
    # in the runner automatically
    source_type = workflow.inputs_spec[source_name]
    if isinstance(source_input, list) and not is_container_type(source_type):
        LOGGER.info(f"Input for source {source_name} is a list, trying to patch workflow")
        try:
            fan_out_workflow_source(workflow, source_name)  # patch is done in-place
        except ValueError:
            raise ValueError(
                f"Found list of inputs for workflow source '{source_name}' "
                f"which does not support lists"
            )


def patch_workflow_sources(user_input: OpIOType, workflow: Workflow):
    bad_sources = []
    for source_name in workflow.inputs_spec:
        source_input = user_input[source_name]
        try:
            patch_workflow_source(source_input, workflow, source_name)
        except ValueError:
            bad_sources.append(source_name)
    if bad_sources:
        raise ValueError(
            f"Found list of inputs for workflow sources {bad_sources} that do not support lists"
        )
