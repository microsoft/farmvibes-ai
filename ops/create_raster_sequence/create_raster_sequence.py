from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

from shapely import geometry as shpg
from shapely import ops as shpo

from vibe_core.data import Raster, gen_guid
from vibe_core.data.rasters import RasterSequence


def get_proper_order(seq: Union[List[Raster], RasterSequence]) -> List[Raster]:
    if isinstance(seq, RasterSequence):
        return [Raster.clone_from(seq, gen_guid(), assets=[i]) for i in seq.get_ordered_assets()]  # type: ignore
    else:
        return sorted(seq, key=lambda r: r.time_range[0])


def get_timerange(list1: List[Raster], list2: List[Raster]) -> Tuple[datetime, datetime]:
    dates = sorted([t for list in [list1, list2] for r in list for t in r.time_range])
    return dates[0], dates[-1]


def get_geom(list1: List[Raster], list2: List[Raster]) -> Dict[str, Any]:
    geoms = [r.geometry for list in [list1, list2] for r in list]
    return shpg.mapping(shpo.unary_union([shpg.shape(i) for i in geoms]))


class CallbackBuilder:
    def __call__(self):
        def create_raster_sequence(
            rasters1: Union[List[Raster], RasterSequence],
            rasters2: Union[List[Raster], RasterSequence],
        ) -> Dict[str, RasterSequence]:
            list1 = get_proper_order(rasters1)
            list2 = get_proper_order(rasters2)

            time_range = get_timerange(list1, list2)
            geom = get_geom(list1, list2)

            res = RasterSequence(
                gen_guid(),
                time_range=time_range,
                geometry=geom,
                assets=[],
                bands=dict(),
            )
            for r in list1:
                res.add_item(r)
            for r in list2:
                res.add_item(r)
            return {"sequence": res}

        return create_raster_sequence
