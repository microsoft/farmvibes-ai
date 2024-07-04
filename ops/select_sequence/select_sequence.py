from functools import partial
from typing import Dict, List, Union

import numpy as np
from shapely.geometry import mapping

from vibe_core.data import Raster, RasterSequence
from vibe_core.data.core_types import gen_guid


def callback(
    rasters: Union[RasterSequence, List[Raster]], num: int, criterion: str
) -> Dict[str, RasterSequence]:
    if isinstance(rasters, RasterSequence):
        rasters = [
            Raster.clone_from(
                rasters,
                gen_guid(),
                assets=[i],
                geometry=mapping(rasters.asset_geometry[i.id]),
                time_range=rasters.asset_time_range[i.id],
            )
            for i in rasters.get_ordered_assets()
        ]

    if len(rasters) < num:
        raise ValueError(
            f"The raster sequence has fewer entries ({len(rasters)}) than requested ({num})"
        )

    if criterion == "first":
        idxs = np.arange(num)
    elif criterion == "last":
        idxs = np.arange(len(rasters) - num, len(rasters))
    elif criterion == "regular":
        idxs = np.round(np.linspace(0, len(rasters) - 1, num)).astype(int)
    else:
        raise ValueError(
            f"Invalid selection criterion {criterion}. "
            f"Valid criteria are 'first', 'last' and 'regular'"
        )

    selected_rasters = [rasters[i] for i in idxs]

    res = RasterSequence.clone_from(rasters[0], f"select_{criterion}_{gen_guid()}", [])

    for r in selected_rasters:
        res.add_item(r)

    return {"sequence": res}


def callback_builder(num: int, criterion: str):
    return partial(callback, num=num, criterion=criterion)
