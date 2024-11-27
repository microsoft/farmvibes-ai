# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Selects a (locally?) minimum subset of items that covers the desired input geometry
(if suchs subset exists) for each timestamp.
Discards items for a timestamp if the geometry cannot be covered at that time.
Assumes items are sparse in time (time range is one moment in time)
"""

from collections import defaultdict
from typing import Dict, List, Optional, TypeVar

from shapely import geometry as shpg
from shapely import ops as shpo
from shapely.geometry.base import BaseGeometry

from vibe_core.data import DataVibe
from vibe_lib.geometry import is_approx_within, norm_intersection

T = TypeVar("T", bound=DataVibe, covariant=True)


def can_cover(geom: BaseGeometry, items: List[T], threshold: float) -> bool:
    item_geoms = [shpg.shape(p.geometry) for p in items]
    return is_approx_within(geom, shpo.unary_union(item_geoms), threshold)


def intersect_area(x: DataVibe, geom: BaseGeometry) -> float:
    return shpg.shape(x.geometry).intersection(geom).area


def filter_necessary_items(
    geom: BaseGeometry, items: List[T], threshold: float, min_area: Optional[float] = None
) -> List[T]:
    """
    Greedily filter the items so that only a subset necessary to cover all
    the geometry's spatial extent is returned
    """
    if min_area is None:
        min_area = (1 - threshold) * geom.area
    if not items:  # No more items left, can't cover the geometry
        return []
    sorted_items = sorted(items, key=lambda x: intersect_area(x, geom), reverse=True)
    # Get item with largest intersection
    item = sorted_items[0]
    item_geom = shpg.shape(item.geometry)
    if is_approx_within(geom, item_geom, threshold):
        return [item]
    if norm_intersection(geom, item_geom) < (1 - threshold):
        # Can't make more progress, so we give up
        return []
    remaining_geom = geom - item_geom
    if remaining_geom.area < min_area:
        # We covered enough of the area, so we stop now
        return [item]
    return [item] + filter_necessary_items(remaining_geom, sorted_items[1:], threshold, min_area)


def callback_builder(
    min_cover: float, within_threshold: float, max_items: Optional[int], group_attribute: str
):
    if not 0 < min_cover < 1:
        raise ValueError(f"{min_cover=} must be between 0 and 1")
    if not 0 < within_threshold < 1:
        raise ValueError(f"{within_threshold=} must be between 0 and 1")
    if min_cover > within_threshold:
        raise ValueError(f"{min_cover=} cannot be larger than {within_threshold}")
    min_cover = min(min_cover, within_threshold)

    def filter_items(bounds_item: DataVibe, items: List[T]) -> Dict[str, T]:
        input_geometry = shpg.shape(bounds_item.geometry)
        item_groups = defaultdict(list)
        for p in items:
            item_groups[getattr(p, group_attribute)].append(p)
        item_groups = [
            sorted(item_group, key=lambda x: intersect_area(x, input_geometry), reverse=True)[
                :max_items
            ]
            for item_group in item_groups.values()
        ]
        filtered_items = {
            item.id: item
            for item_group in item_groups
            if can_cover(
                input_geometry,
                item_group,
                min_cover,
            )
            for item in filter_necessary_items(input_geometry, item_group, within_threshold)
        }
        if not filtered_items:
            raise RuntimeError(f"No product group can cover input geometry {bounds_item.geometry}")
        return filtered_items

    def callback(bounds_items: List[DataVibe], items: List[T]) -> Dict[str, List[T]]:
        filtered_items = {}
        for bounds_item in bounds_items:
            filtered_items.update(filter_items(bounds_item, items))

        return {"filtered_items": [v for v in filtered_items.values()]}

    return callback
