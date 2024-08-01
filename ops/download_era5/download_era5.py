# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import mimetypes
import os
from tempfile import TemporaryDirectory
from typing import Dict, Optional, cast

import cdsapi
import fsspec
import planetary_computer as pc
import xarray as xr

from vibe_core.data import AssetVibe, Era5Product, gen_guid, gen_hash_id
from vibe_lib.planetary_computer import Era5Collection

LOGGER = logging.getLogger(__name__)


class CallbackBuilder:
    def __init__(self, api_key: str):
        self.tmp_dir = TemporaryDirectory()
        self.api_key = api_key

    def __call__(self):
        def download_product(
            era5_product: Era5Product,
        ) -> Dict[str, Optional[Era5Product]]:
            if era5_product.item_id != "":
                pc.set_subscription_key(self.api_key)
                collection = Era5Collection()
                item = collection.query_by_id(era5_product.item_id)

                # Only downloading the asset corresponding to the requested variable.
                # In addition, the requested asset is a zarr, which is a directory structure,
                # so it not possible to use download_asset.
                signed_item = pc.sign(item)
                asset = signed_item.assets[era5_product.var]
                ds = xr.open_dataset(asset.href, **asset.extra_fields["xarray:open_kwargs"])
            else:
                if self.api_key == "":
                    raise ValueError(
                        "api_key not supplied for CDS (registration "
                        "in https://cds.climate.copernicus.eu/user/register)"
                    )
                if len(era5_product.cds_request) != 1:
                    raise ValueError(f"Invalid number of CDS requests {era5_product.cds_request}")
                dataset, request = next((k, v) for k, v in era5_product.cds_request.items())
                c = cdsapi.Client(url="https://cds.climate.copernicus.eu/api/v2", key=self.api_key)
                r = c.retrieve(dataset, request)
                if r is None:
                    raise ValueError(f"CDS request {era5_product.cds_request} returned None")
                with fsspec.open(r.location) as f:
                    ds = xr.open_dataset(f, engine="scipy")  # type: ignore

            path = os.path.join(self.tmp_dir.name, f"{era5_product.id}.nc")
            ds.to_netcdf(path)
            vibe_asset = AssetVibe(
                reference=path, type=cast(str, mimetypes.guess_type(path)[0]), id=gen_guid()
            )
            downloaded_product = Era5Product.clone_from(
                era5_product,
                id=gen_hash_id(
                    f"{era5_product.id}_downloaded", era5_product.geometry, era5_product.time_range
                ),
                assets=[vibe_asset],
            )

            return {"downloaded_product": downloaded_product}

        return download_product

    def __del__(self):
        self.tmp_dir.cleanup()
