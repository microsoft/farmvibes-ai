from tempfile import TemporaryDirectory
from typing import Dict

import xarray as xr

from vibe_core.data import Sentinel2CloudProbability, gen_guid
from vibe_lib.raster import load_raster, save_raster_to_asset


class CallbackBuilder:
    def __init__(self) -> None:
        self.tmp_dir = TemporaryDirectory()

    def __call__(self):
        def ensemble_cloud_prob(
            cloud1: Sentinel2CloudProbability,
            cloud2: Sentinel2CloudProbability,
            cloud3: Sentinel2CloudProbability,
            cloud4: Sentinel2CloudProbability,
            cloud5: Sentinel2CloudProbability,
        ) -> Dict[str, Sentinel2CloudProbability]:
            ar = [load_raster(c) for c in (cloud1, cloud2, cloud3, cloud4, cloud5)]
            ar = xr.concat(ar, dim="band").mean(dim="band")
            asset = save_raster_to_asset(ar, self.tmp_dir.name)
            return {
                "cloud_probability": Sentinel2CloudProbability.clone_from(
                    cloud1, id=gen_guid(), assets=[asset]
                )
            }

        return ensemble_cloud_prob
