# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random
from typing import Dict, List

import pytest

from vibe_server.workflow.graph import Graph


class SomeGraph(Graph[int, int]):
    def __init__(self, data: Dict[int, List[int]]):
        super().__init__()

        for k in data:
            self.add_node(k)
        for k, v in data.items():
            for d in v:
                self.add_edge(k, d, 1)


@pytest.fixture
def loopy_graph() -> SomeGraph:
    return SomeGraph(
        {
            0: [1, 2, 3],
            1: [2, 3, 4],
            2: [3],
            3: [4],
            4: [3],
            5: [2],
        }
    )


@pytest.fixture
def a_normal_graph() -> SomeGraph:
    # topological sort: [0], [1, 4], [2, 5, 6, 7], [3]
    # graph:
    #                  /-> 7
    #                 /-> 6
    #            /-> 4 -> 5
    #          0 -> 1 -> 2 -> 3
    #           \-------/    /
    #            \---------/
    #
    return SomeGraph(
        {
            0: [1, 2, 3, 4],
            1: [2, 3],
            2: [3],
            3: [],
            4: [5, 6, 7],
            5: [],
            6: [],
            7: [],
        }
    )


@pytest.fixture
def a_simple_graph() -> SomeGraph:
    #           /-> 🔙 \
    # 🌎 -> 🎶 --> 🔚  \-> ✅
    #
    return SomeGraph(
        {
            int.from_bytes("🌎".encode("utf-8"), "little"): [
                int.from_bytes("🎶".encode("utf-8"), "little")
            ],
            int.from_bytes("🎶".encode("utf-8"), "little"): [
                int.from_bytes("🔙".encode("utf-8"), "little"),
                int.from_bytes("🔚".encode("utf-8"), "little"),
            ],
            int.from_bytes("🔙".encode("utf-8"), "little"): [
                int.from_bytes("✅".encode("utf-8"), "little")
            ],
            int.from_bytes("🔚".encode("utf-8"), "little"): [
                int.from_bytes("✅".encode("utf-8"), "little")
            ],
        }
    )


@pytest.fixture
def empty_graph() -> SomeGraph:
    return SomeGraph({})


def test_topological_sort_on_empty_graph(empty_graph: SomeGraph):
    assert list(empty_graph.topological_sort()) == []


def test_cycle_detection_on_empty_graph(empty_graph: SomeGraph):
    assert not empty_graph.has_cycle()


def test_loopy_graph_has_cycle(loopy_graph: SomeGraph):
    assert loopy_graph.has_cycle()


def test_topological_sort_on_a_loopy_graph(loopy_graph: SomeGraph):
    with pytest.raises(ValueError):
        loopy_graph.topological_sort()


def test_topological_sort_on_a_normal_graph(a_normal_graph: SomeGraph):
    sort = list(a_normal_graph.topological_sort())
    assert sort[0] == [0]
    assert sort[1] == [1, 4]
    assert sort[2] == [2, 5, 6, 7]
    assert sort[3] == [3]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_topological_sort_on_a_simple_graph(a_simple_graph: SomeGraph):
    sort = list(a_simple_graph.topological_sort())
    assert sort[0] == [int.from_bytes("🌎".encode("utf-8"), "little")]
    assert sort[1] == [int.from_bytes("🎶".encode("utf-8"), "little")]
    assert set(sort[2]) == set(
        [
            int.from_bytes("🔙".encode("utf-8"), "little"),
            int.from_bytes("🔚".encode("utf-8"), "little"),
        ]
    )
    assert sort[3] == [int.from_bytes("✅".encode("utf-8"), "little")]


def test_topological_sort_on_random_graphs():
    with pytest.warns(UserWarning):
        for _ in range(42):
            a = random.randint(-999999, 999999)
            b = random.randint(-999999, 999999)
            c = random.randint(-999999, 999999)
            graph = SomeGraph({a: [b, c], b: [c]})
            sort = list(graph.topological_sort())
            assert len(sort) == 3
            assert sort[0] == [a]
            assert sort[1] == [b]
            assert sort[2] == [c]


def test_relabel_normal_graph(a_normal_graph: SomeGraph):
    edge1 = (1, 2, 1)
    a_normal_graph.relabel(edge1, 2)
    assert 2 in a_normal_graph.neighbors(1)
    assert (2, 2) in a_normal_graph.adjacency_list[1]
    assert (2, 1) not in a_normal_graph.adjacency_list[1]


def test_no_relabel_missing_edge(a_normal_graph: SomeGraph):
    edge = (3, 4, 1)
    with pytest.raises(KeyError):
        a_normal_graph.relabel(edge, 2)
