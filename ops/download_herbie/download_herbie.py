from datetime import timedelta
from tempfile import TemporaryDirectory
from typing import Dict, Optional

import rasterio
from herbie import Herbie

from vibe_core.data import AssetVibe, Grib
from vibe_core.data.core_types import gen_guid
from vibe_core.data.products import HerbieProduct


class CallbackBuilder:
    def __init__(self):
        self.tmp_dir = TemporaryDirectory()

    def __call__(self):
        def download_herbie(
            herbie_product: HerbieProduct,
        ) -> Dict[str, Optional[Grib]]:
            H = Herbie(
                herbie_product.time_range[0].replace(tzinfo=None),
                fxx=herbie_product.lead_time_hours,
                model=herbie_product.model,
                product=herbie_product.product,
            )
            grib_path = H.download(herbie_product.search_text)
            asset = AssetVibe(reference=str(grib_path), type="application/x-grib", id=gen_guid())
            with rasterio.open(grib_path) as f:
                t = herbie_product.time_range[0] + timedelta(hours=herbie_product.lead_time_hours)
                forecast = Grib.clone_from(
                    herbie_product,
                    time_range=(t, t),
                    id=gen_guid(),
                    assets=[asset],
                    meta={"lead_time": str(herbie_product.lead_time_hours)},
                    bands={
                        f.tags(i)["GRIB_ELEMENT"]: i - 1  # type: ignore
                        for i in range(1, f.count + 1)  # type: ignore
                    },
                )

            return {"forecast": forecast}

        return download_herbie

    def __del__(self):
        self.tmp_dir.cleanup()
