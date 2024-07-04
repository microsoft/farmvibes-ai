import hashlib
from collections import defaultdict
from typing import Dict, List, Tuple, Union, cast

from shapely import geometry as shpg
from shapely.ops import unary_union

from vibe_core.data.sentinel import (
    Sentinel2CloudMask,
    Sentinel2CloudMaskOrbitGroup,
    Sentinel2Raster,
    Sentinel2RasterOrbitGroup,
    discriminator_date,
)
from vibe_lib.spaceeye.utils import find_s2_product

TileData = List[Tuple[Sentinel2Raster, Sentinel2CloudMask]]


def make_orbit_group(
    items: TileData,
) -> Tuple[Sentinel2RasterOrbitGroup, Sentinel2CloudMaskOrbitGroup]:
    # Make sure we are ordered by time make things consistent for the id hash
    rasters, masks = zip(*sorted(items, key=lambda x: discriminator_date(x[0].product_name)))
    rasters = cast(List[Sentinel2Raster], list(rasters))
    masks = cast(List[Sentinel2CloudMask], list(masks))
    # Id depends on all component ids
    raster_group_id, cloud_group_id = [
        hashlib.sha256("".join(i.id for i in items).encode()).hexdigest()
        for items in (rasters, masks)
    ]
    geom = shpg.mapping(unary_union([shpg.shape(r.geometry) for r in rasters]))
    # dates = [r.time_range[0] for r in rasters]
    # time_range = (min(dates), max(dates))
    raster_group = Sentinel2RasterOrbitGroup.clone_from(
        rasters[-1], id=raster_group_id, assets=[], geometry=geom
    )
    for r in rasters:
        raster_group.add_raster(r)
    mask_group = Sentinel2CloudMaskOrbitGroup.clone_from(
        masks[-1], id=cloud_group_id, assets=[], geometry=geom
    )
    for m in masks:
        mask_group.add_raster(m)
    return raster_group, mask_group


def callback_builder():
    def group_by_orbit(
        rasters: List[Sentinel2Raster],
        masks: List[Sentinel2CloudMask],
    ) -> Dict[str, Union[List[Sentinel2RasterOrbitGroup], List[Sentinel2CloudMaskOrbitGroup]]]:
        same_orbits: Dict[Tuple[int, str], TileData] = defaultdict(list)
        for item in rasters:
            orbit_key = (item.orbit_number, item.tile_id)
            mask_item = find_s2_product(item.product_name, masks)
            same_orbits[orbit_key].append((item, mask_item))

        groups = [make_orbit_group(v) for v in same_orbits.values()]
        raster_groups, mask_groups = zip(*groups)
        raster_groups = cast(List[Sentinel2RasterOrbitGroup], list(raster_groups))
        mask_groups = cast(List[Sentinel2CloudMaskOrbitGroup], list(mask_groups))

        return {"raster_groups": raster_groups, "mask_groups": mask_groups}

    return group_by_orbit
