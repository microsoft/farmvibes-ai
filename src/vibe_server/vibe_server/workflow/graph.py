#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from collections import defaultdict
from enum import IntEnum
from typing import Callable, Dict, Generic, Iterable, Iterator, List, Set, Tuple, TypeVar
from warnings import warn

T = TypeVar("T")
V = TypeVar("V")
Edge = Tuple[T, T, V]
Adjacency = Set[Tuple[T, V]]


class VisitStatus(IntEnum):
    new = 0
    visiting = 1
    visited = 2


class Graph(Generic[T, V]):
    adjacency_list: Dict[T, Adjacency[T, V]]

    def __init__(self):
        self.adjacency_list = {}

    def add_node(self, node: T):
        if node in self.adjacency_list:
            warn(f"Trying to add already existing node {node} to graph. Ignoring.")
        else:
            self.adjacency_list[node] = set()

    def add_edge(self, origin: T, destination: T, label: V):
        if origin not in self.adjacency_list:
            warn(f"Tried to add edge from {origin} to {destination}, but {origin} not in graph")
            self.add_node(origin)
        if destination not in self.adjacency_list:
            warn(
                f"Tried to add edge from {origin} to {destination}, but {destination} not in graph"
            )
            self.add_node(destination)
        self.adjacency_list[origin].add((destination, label))

    def relabel(self, edge: Edge[T, V], new_label: V):
        """Changes an existing edge's label to `new_label`."""
        self.adjacency_list[edge[0]].remove((edge[1], edge[2]))
        self.adjacency_list[edge[0]].add((edge[1], new_label))

    @property
    def nodes(self) -> List[T]:
        return list(self.adjacency_list.keys())

    @property
    def edges(self) -> List[Edge[T, V]]:
        return [
            (origin, destination[0], destination[1])
            for origin, destinations in self.adjacency_list.items()
            for destination in destinations
        ]

    def neighbors(self, vertex: T) -> Set[T]:
        return set(e[0] for e in self.adjacency_list[vertex])

    def edges_from(self, vertex: T) -> Iterable[Edge[T, V]]:
        return [(vertex, *dst) for dst in self.adjacency_list[vertex]]

    def zero_in_degree_nodes(self) -> Iterable[T]:
        in_degrees: Dict[T, int] = {k: 0 for k in self.adjacency_list}
        for destinations in self.adjacency_list.values():
            for destination in destinations:
                in_degrees[destination[0]] += 1
        return [k for k, v in in_degrees.items() if v == 0]

    def _dfs_impl(
        self,
        vertex: T,
        visited: Dict[T, Tuple[VisitStatus, int]],
        level: int = 0,
        visit: Callable[[int, T, VisitStatus], None] = lambda i, v, s: None,
    ) -> None:
        if len(visited) == 0:
            for v in self.nodes:
                visited[v] = (VisitStatus.new, 0)

        if visited[vertex][0] == VisitStatus.visited and level < visited[vertex][1]:
            return

        visit(level, vertex, VisitStatus.visiting)
        for neighbor in self.neighbors(vertex):
            try:
                if visited[neighbor][0] == VisitStatus.visiting:
                    raise ValueError(f"Graph has a cycle with at least node {neighbor}")
                elif visited[neighbor][0] == VisitStatus.new or (level + 1 > visited[neighbor][1]):
                    # Haven't visited this, or need to revisit at a higher level
                    self._dfs_impl(neighbor, visited, level + 1, visit)
            except KeyError:
                # We just reached a node we didn't even know existed
                # This is probably a terminal node
                warn(f"Found node {neighbor}, but it is not in the list of nodes.")
                self._dfs_impl(neighbor, visited, level + 1, visit)

        visit(level, vertex, VisitStatus.visited)

    def has_cycle(self) -> bool:
        try:
            self.topological_sort()
            return False
        except ValueError as e:
            if "cycle" in str(e):
                return True
            raise

    def topological_sort(self) -> Iterable[List[T]]:
        """Performs topological sort in a graph.

        Returns an iterable for all connected components. Raises exception if
        the graph has a cycle.
        """
        visited: Dict[T, Tuple[VisitStatus, int]] = {k: (VisitStatus.new, 0) for k in self.nodes}

        def visit(level: int, vertex: T, status: VisitStatus):
            visited[vertex] = status, level

        for source in self.zero_in_degree_nodes():
            assert visited[source][0] == VisitStatus.new, f"Visited source {source} more than once"
            visit(0, source, VisitStatus.visiting)
            for neighbor in self.neighbors(source):
                self._dfs_impl(neighbor, level=1, visit=visit, visited=visited)
            visit(0, source, VisitStatus.visited)
        if not all([v[0] == VisitStatus.visited for v in visited.values()]):
            raise ValueError(
                "Not all nodes visited in topological sort. This indicates "
                "disconnected components in the graph."
            )

        ordering: Dict[int, List[T]] = defaultdict(list)
        for node, (_, level) in visited.items():
            ordering[level].append(node)
        return (ordering[k] for k in sorted(ordering.keys()))

    def __iter__(self) -> Iterator[List[T]]:
        return (v for v in self.topological_sort())
