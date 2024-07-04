import hashlib
from collections import defaultdict
from typing import Dict, List, Tuple

from shapely import geometry as shpg
from shapely.ops import unary_union

from vibe_core.data import Sentinel1Raster, Sentinel1RasterOrbitGroup


def make_orbit_group(
    items: List[Sentinel1Raster],
) -> Sentinel1RasterOrbitGroup:
    # Make sure we are ordered by time make things consistent for the id hash
    rasters = sorted(items, key=lambda x: x.time_range[0])
    # Id depends on all component ids
    group_id = hashlib.sha256("".join(i.id for i in rasters).encode()).hexdigest()
    geom = shpg.mapping(unary_union([shpg.shape(r.geometry) for r in rasters]))
    dates = [r.time_range[0] for r in rasters]
    time_range = (min(dates), max(dates))
    group = Sentinel1RasterOrbitGroup.clone_from(
        rasters[0], id=group_id, assets=[], time_range=time_range, geometry=geom
    )
    for r in rasters:
        group.add_raster(r)

    return group


def callback_builder():
    def group_by_orbit(
        rasters: List[Sentinel1Raster],
    ) -> Dict[str, List[Sentinel1RasterOrbitGroup]]:
        same_orbits: Dict[Tuple[int, str], List[Sentinel1Raster]] = defaultdict(list)
        for item in rasters:
            orbit_key = (item.orbit_number, item.tile_id)
            same_orbits[orbit_key].append(item)

        groups = [make_orbit_group(v) for v in same_orbits.values()]

        return {"raster_groups": groups}

    return group_by_orbit
