import hashlib
from enum import auto
from typing import Dict, List, TypeVar

from shapely import geometry as shpg
from shapely import ops as shpo
from strenum import StrEnum

from vibe_core.data import DataVibe

T = TypeVar("T", bound=DataVibe)


class MergeMethod(StrEnum):
    union = auto()
    intersection = auto()


def callback_builder(method: str):
    try:
        merge_method = MergeMethod[method]
    except KeyError:
        avail_methods = ", ".join([i.name for i in MergeMethod])
        raise ValueError(
            f"Invalid merge method parameter {method}. Available methods are {avail_methods}"
        )

    def callback(items: List[T]) -> Dict[str, T]:
        item_type = type(items[0])

        if merge_method == MergeMethod.union:
            merge_geom = shpg.mapping(shpo.unary_union([shpg.shape(i.geometry) for i in items]))
        else:
            merge_geom = shpg.shape(items[0].geometry)
            for i in items:
                merge_geom = merge_geom.intersection(shpg.shape(i.geometry))
            merge_geom = shpg.mapping(merge_geom)
        merge_id = hashlib.sha256(
            "".join([f"merge geometries method={merge_method}"] + [i.id for i in items]).encode()
        ).hexdigest()
        return {
            "merged": item_type.clone_from(items[0], id=merge_id, assets=[], geometry=merge_geom)
        }

    return callback
