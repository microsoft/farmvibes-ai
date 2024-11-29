# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""General utility functions used across FarmVibes.AI codebase."""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, TypeVar, Union

from vibe_core.data.core_types import OpIOType

T = TypeVar("T")


@dataclass
class MermaidVerticesMap:
    """Map of vertices for a mermaid diagram extracted from a WorkflowSpec.

    Each entry maps the source/sink/task name to the vertex label.
    """

    sources: Dict[str, str]
    """Source map."""

    sinks: Dict[str, str]
    """Sink map."""

    tasks: Dict[str, str]
    """Task map."""


def ensure_list(input: Union[List[T], T]) -> List[T]:
    """Ensure that the given input is a list.

    If the input is a single item, it is wrapped in a list.

    Args:
        input: List or single item to be wrapped in a list.

    Returns:
        A list containing the input item.
    """
    if isinstance(input, list):
        return input
    return [input]


def get_input_ids(input: OpIOType) -> Dict[str, Union[str, List[str]]]:
    """Retrieve the IDs from an input OpIOType object.

    This method will extract the IDs from an OpIOType object and return them as a dictionary,
    where the keys are the names of the inputs and values are either strings or lists of strings.

    Args:
        input: The input object.

    Returns:
        A dictionary with the IDs of the input object.
    """
    return {
        k: [vv.get("id", "NO-ID") for vv in v] if isinstance(v, list) else v.get("id", "NO-ID")
        for k, v in input.items()
    }


def rename_keys(x: Dict[str, Any], key_dict: Dict[str, str]):
    """Rename the keys of a dictionary.

    This utility function takes a dictionary `x` and a dictionary `key_dict`
    mapping old keys to their new names, and returns a copy of `x` with the keys renamed.

    Args:
        x: The dictionary with the keys to be renamed.
        key_dict: Dictionary mapping old keys to their new names.

    Returns:
        A copy of x with the keys renamed.
    """
    renamed = x.copy()
    for old_key, new_key in key_dict.items():
        if old_key in x:
            renamed[new_key] = x[old_key]
            del renamed[old_key]
    return renamed


def format_double_escaped(s: str):
    """Encode and decode a double escaped input string.

    Useful for formatting status/reason strings of VibeWorkflowRun.

    Args:
        s: Input string to be processed.

    Returns:
        Formatted string.
    """
    return s.encode("raw_unicode_escape").decode("unicode-escape")


def build_mermaid_edge(
    origin: Tuple[str, str],
    destination: Tuple[str, str],
    vertices_origin: Dict[str, str],
    vertices_destination: Dict[str, str],
) -> str:
    """Build a mermaid edge from a pair of vertices.

    Args:
        origin: A pair of source/sink/task and port names.
        destination: A pair of source/sink/task and port names.
        vertices_origin: The vertex map to retrieve the mermaid vertex label for the origin.
        vertices_destination: The vertex map to retrieve the mermaid vertex label
            for the destination.

    Returns:
        The mermaid edge string.
    """
    origin_vertex, origin_port = origin
    destination_vertex, destination_port = destination

    separator = "/" if origin_port and destination_port else ""

    if origin_port == destination_port:
        port_map = origin_port
    else:
        port_map = f"{origin_port}{separator}{destination_port}"
    return (
        f"{vertices_origin[origin_vertex]} "
        f"-- {port_map} --> "
        f"{vertices_destination[destination_vertex]}"
    )


def draw_mermaid_diagram(vertices: MermaidVerticesMap, edges: List[str]) -> str:
    """Draw a mermaid diagram from a set of vertices and edges.

    Args:
        vertices: A map of vertices for a mermaid diagram extracted from a WorkflowSpec.
        edges: A list of edges already formated with mermaid syntax.

    Returns:
        The mermaid diagram string.
    """
    diagram = (
        "graph TD\n"
        + "\n".join(
            [f"    {source}" for source in vertices.sources.values()]
            + [f"    {sink}" for sink in vertices.sinks.values()]
            + [f"    {task}" for task in vertices.tasks.values()]
        )
        + "\n"
        + "\n".join([f"    {edge}" for edge in edges])
    )

    return diagram
