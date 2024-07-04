import hashlib
from functools import partial
from typing import Dict, List

from shapely import geometry as shpg

from vibe_core.data import DataVibe, Raster, RasterSequence
from vibe_lib.geometry import is_approx_equal


def callback(
    rasters: List[Raster], group_by: List[DataVibe], threshold: float
) -> Dict[str, List[RasterSequence]]:
    ref_bands = rasters[0].bands
    if not all(r.bands == ref_bands for r in rasters):
        raise ValueError("Expected to group rasters with the same bands")
    sequences: List[RasterSequence] = []
    for g in group_by:
        matching_rasters: List[Raster] = []
        geom_g = shpg.shape(g.geometry)
        for r in rasters:
            geom_r = shpg.shape(r.geometry)
            if is_approx_equal(geom_r, geom_g, threshold=threshold):
                matching_rasters.append(r)
        matching_rasters = sorted(matching_rasters, key=lambda x: x.id)
        t = [r.time_range[0] for r in matching_rasters]
        seq = RasterSequence(
            id=hashlib.sha256("".join([r.id for r in matching_rasters]).encode()).hexdigest(),
            time_range=(min(t), max(t)),
            geometry=g.geometry,
            assets=[],
            bands=ref_bands,
        )
        for r in matching_rasters:
            seq.add_item(r)
        sequences.append(seq)
    return {"raster_groups": sequences}


def callback_builder(geom_threshold: float):
    return partial(callback, threshold=geom_threshold)
