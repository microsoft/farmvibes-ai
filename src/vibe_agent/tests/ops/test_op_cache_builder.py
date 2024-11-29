# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import datetime
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from pystac import Item
from pytest import fixture
from shapely.geometry import Polygon, mapping

from vibe_common.schemas import CacheInfo, ItemDict, OpResolvedDependencies
from vibe_core.data.core_types import BaseVibe


@dataclass
class TestVibe(BaseVibe):
    a: int
    b: str


@fixture
def item_dict():
    num_items = 5
    polygon_coords = [
        (-88.062073563448919, 37.081397673802059),
        (-88.026349330507315, 37.085463858128762),
        (-88.026349330507315, 37.085463858128762),
        (-88.012445388773259, 37.069230099135126),
    ]
    polygon: Dict[str, Any] = mapping(Polygon(polygon_coords))  # type: ignore
    timestamp = datetime.datetime.now(datetime.timezone.utc)
    items = [
        Item(id=str(i), geometry=polygon, datetime=timestamp, properties={}, bbox=None)
        for i in range(num_items)
    ]
    single_item = Item(
        id=str(num_items), geometry=polygon, datetime=timestamp, properties={}, bbox=None
    )
    base_items = [TestVibe(i, f"{i}") for i in range(num_items)]
    single_base = TestVibe(num_items, f"{num_items}")

    output_dict = {
        "list_input": items,
        "single_input": single_item,
        "list_base": base_items,
        "single_base": single_base,
    }

    return output_dict


@fixture
def item_dict_hashes() -> Dict[str, Union[str, List[str], Dict[str, Any]]]:
    return {
        "vibe_source_items": {
            "list_input": ["0", "1", "2", "3", "4"],
            "single_input": "5",
            "list_base": [
                "371c8cb9ac0a9f7d31fd0ab9d1e59efe3a5d98854e86b6bfa3207ccf4e6dfbf6",
                "3d15b923441e57a7b3f9dcc93f43d8b41620b3dba7d5c4be78bf0b2a597006d2",
                "c5e1ca033cc639402b7352606e8a00676636287f437739a1c773440df76d2799",
                "cf3b5755718f90ffe7cdf7b27bd41da19158ea4d1fefdc7aca188bc9dcac7f19",
                "eab1e3a83e5b227da228fefdf633ce9a05b12dcdb59d6739f7d1dddeb51d712f",
            ],
            "single_base": "66756d10b406f729019b8a049f02e293b7f7e0e3b22f613f4c7024f732e5ee11",
        },
        "vibe_op_parameters": {"parameters": {"dep": 1, "another_dep": "bla"}},
        "vibe_op_version": "1",
        "vibe_op_hash": "5daf389eaad4c50533c2b1ace0b6f551f1a3b9236ec35f1fa3e5a5ab11b68a32",
    }


@fixture
def op_dependencies():
    return {"parameters": {"dep": 1, "another_dep": "bla"}}


def test_stable_hashes(
    item_dict: ItemDict,
    op_dependencies: OpResolvedDependencies,
    item_dict_hashes: Dict[str, Union[str, List[str], Dict[str, Any]]],
):
    cache_info = CacheInfo("test_op", "1.0", item_dict, op_dependencies)
    storage_dict = cache_info.as_storage_dict()
    for k, v in item_dict_hashes.items():
        assert storage_dict[k] == v


def test_cache_builder(item_dict: ItemDict, op_dependencies: OpResolvedDependencies):
    version = "1.3"
    cache_info = CacheInfo("test_op", version, item_dict, op_dependencies)

    assert cache_info.version == version[0]

    for k, v in item_dict.items():
        if isinstance(v, list):
            target_ids = sorted(CacheInfo._compute_or_extract_id(v))
            for target_id, input_id in zip(target_ids, cache_info.ids[k]):
                assert target_id == input_id
        else:
            assert cache_info.ids[k] == CacheInfo._compute_or_extract_id(v)


def test_cache_builder_hash(item_dict: ItemDict, op_dependencies: OpResolvedDependencies):
    version = "1.3"
    cache_info = CacheInfo("test_op", version, item_dict, op_dependencies)
    cache_info_repeat = CacheInfo("test_op", version[0], item_dict, op_dependencies)

    assert cache_info.hash == cache_info_repeat.hash


def test_hash_order_invariances(item_dict: ItemDict, op_dependencies: OpResolvedDependencies):
    version = "1.3"
    cache_info = CacheInfo("test_op", version, item_dict, op_dependencies)

    # Shufling input ids
    random.shuffle(item_dict["list_input"])  # type: ignore
    random.shuffle(item_dict["list_base"])  # type: ignore
    cache_info_shuffled = CacheInfo("test_op", version, item_dict, op_dependencies)

    assert cache_info.hash == cache_info_shuffled.hash


def test_hash_version_dependency(item_dict: ItemDict, op_dependencies: OpResolvedDependencies):
    cache_info = CacheInfo("test_op", "1.3", item_dict, op_dependencies)
    cache_info_repeat = CacheInfo("test_op", "2.5", item_dict, op_dependencies)

    assert cache_info.hash != cache_info_repeat.hash


def test_hash_source_id_dependency_single(
    item_dict: ItemDict, op_dependencies: OpResolvedDependencies
):
    cache_info = CacheInfo("test_op", "1.3", item_dict, op_dependencies)
    item_dict["single_input"].id = str(10)  # type: ignore
    cache_info2 = CacheInfo("test_op", "1.3", item_dict, op_dependencies)
    item_dict["single_base"].a = 2  # type: ignore
    cache_info3 = CacheInfo("test_op", "1.3", item_dict, op_dependencies)

    assert cache_info.hash != cache_info2.hash
    assert cache_info.hash != cache_info3.hash
    assert cache_info2.hash != cache_info3.hash


def test_hash_source_id_dependency_list(
    item_dict: ItemDict, op_dependencies: OpResolvedDependencies
):
    cache_info = CacheInfo("test_op", "1.3", item_dict, op_dependencies)
    item_dict["list_input"][-1].id = str(10)  # type: ignore
    cache_info2 = CacheInfo("test_op", "1.3", item_dict, op_dependencies)
    item_dict["list_base"][-1].b = str(10)  # type: ignore
    cache_info3 = CacheInfo("test_op", "1.3", item_dict, op_dependencies)

    assert cache_info.hash != cache_info2.hash
    assert cache_info.hash != cache_info3.hash
    assert cache_info2.hash != cache_info3.hash


def test_hash_source_name_dependency(item_dict: ItemDict, op_dependencies: OpResolvedDependencies):
    cache_info = CacheInfo("test_op", "1.3", item_dict, op_dependencies)
    i = item_dict.pop("list_input")
    item_dict["different_name_input"] = i
    cache_info_repeat = CacheInfo("test_op", "1.3", item_dict, op_dependencies)

    assert cache_info.hash != cache_info_repeat.hash


def test_hash_parameter_dependency(item_dict: ItemDict, op_dependencies: OpResolvedDependencies):
    op_version = "1.3"
    cache_info = CacheInfo("test_op", op_version, item_dict, op_dependencies)
    op_dependencies["parameters"]["dep"] = 2
    cache_info_repeat = CacheInfo("test_op", op_version, item_dict, op_dependencies)

    assert cache_info.hash != cache_info_repeat.hash


def test_hash_gen_basevibe():
    x = CacheInfo._compute_or_extract_id(TestVibe(1, "1"))
    y = CacheInfo._compute_or_extract_id(TestVibe(2, "1"))
    z = CacheInfo._compute_or_extract_id(TestVibe(1, "2"))
    assert x != y
    assert x != z
    assert y != z
