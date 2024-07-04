import os
from datetime import datetime
from tempfile import TemporaryDirectory
from typing import List, Tuple, cast

import numpy as np
import pytest
import rioxarray as rio
import spyndex
import xarray as xr
from index import compute_methane, compute_ndre, compute_reci
from shapely import geometry as shpg

from vibe_core.data import Raster
from vibe_dev.testing.op_tester import OpTester
from vibe_lib.raster import save_raster_to_asset

YAML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compute_index.yaml")


# code originally on index.py. now we are using spyndex
def compute_ndvi(bands: xr.DataArray) -> xr.DataArray:
    red, nir = bands
    ndvi: xr.DataArray = (nir - red) / (nir + red)
    ndvi.rio.write_nodata(100, encoded=True, inplace=True)
    return ndvi


# code originally on index.py. now we are using spyndex
def compute_evi(bands: xr.DataArray) -> xr.DataArray:
    blue, red, nir = bands
    evi: xr.DataArray = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
    evi.rio.write_nodata(100, encoded=True, inplace=True)
    return evi


# code originally on index.py. now we are using spyndex
def compute_msavi(bands: xr.DataArray) -> xr.DataArray:
    """
    Modified Soil Adjusted Vegetation Index.
    This is technically MSAVI_2 which is frequently used as MSAVI
    """
    red, nir = bands
    disc = (2 * nir + 1) ** 2 - 8 * (nir - red)
    msavi: xr.DataArray = (2 * nir + 1 - disc**0.5) / 2.0
    msavi.rio.write_nodata(100, encoded=True, inplace=True)
    return msavi


# code originally on index.py. now we are using spyndex
def compute_ndmi(bands: xr.DataArray) -> xr.DataArray:
    """
    Normalized Difference Moisture Index
    """
    nir, swir16 = bands
    ndmi: xr.DataArray = (nir - swir16) / (nir + swir16)
    ndmi.rio.write_nodata(100, encoded=True, inplace=True)
    return ndmi


def compute_ndwi(bands: xr.DataArray) -> xr.DataArray:
    g, n = bands
    return spyndex.indices.NDWI.compute(G=g, N=n)


def compute_lswi(bands: xr.DataArray) -> xr.DataArray:
    n, s1 = bands
    return spyndex.indices.LSWI.compute(N=n, S1=s1)


def compute_nbr(bands: xr.DataArray) -> xr.DataArray:
    n, s2 = bands
    return spyndex.indices.NBR.compute(N=n, S2=s2)


true_index_fn = {
    "ndvi": compute_ndvi,
    "evi": compute_evi,
    "msavi": compute_msavi,
    "ndmi": compute_ndmi,
    "ndwi": compute_ndwi,
    "methane": compute_methane,
    "ndre": compute_ndre,
    "reci": compute_reci,
    "LSWI": compute_lswi,
    "NBR": compute_nbr,
}


def create_fake_raster(
    tmp_dir_name: str, bands: List[str], y: int, x: int
) -> Tuple[Raster, xr.DataArray]:
    nbands = len(bands)
    fake_data = np.random.random((nbands, y, x)).astype(np.float32)
    fake_da = xr.DataArray(
        fake_data,
        coords={"bands": np.arange(nbands), "x": np.linspace(0, 1, x), "y": np.linspace(0, 1, y)},
        dims=["bands", "y", "x"],
    )
    fake_da.rio.write_crs("epsg:4326", inplace=True)

    asset = save_raster_to_asset(fake_da, tmp_dir_name)

    return (
        Raster(
            id="fake_id",
            time_range=(datetime(2023, 1, 1), datetime(2023, 1, 1)),
            geometry=shpg.mapping(shpg.box(*fake_da.rio.bounds())),
            assets=[asset],
            bands={j: i for i, j in enumerate(bands)},
        ),
        fake_da,
    )


@pytest.fixture
def tmp_dir():
    _tmp_dir = TemporaryDirectory()
    yield _tmp_dir.name
    _tmp_dir.cleanup()


@pytest.mark.parametrize(
    "bands, index, should_fail",
    [
        (["R", "N"], "ndvi", False),
        (["B", "R", "N"], "evi", False),
        (["R", "N"], "msavi", False),
        (["N", "S1"], "ndmi", False),
        (["RE1", "N"], "ndre", False),
        (["RE1", "N"], "reci", False),
        (
            ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B12"],
            "methane",
            False,
        ),
        (["G", "N"], "ndwi", False),
        (["N"], "LSWI", True),
        (["N", "S1"], "LSWI", False),
        (["N", "S2"], "NBR", False),
    ],
)
def test_op(bands: List[str], index: str, should_fail: bool, tmp_dir: str):
    raster, da = create_fake_raster(tmp_dir, bands, 20, 20)
    op_tester = OpTester(YAML_PATH)
    parameters = {"index": index}
    op_tester.update_parameters(parameters)
    try:
        output = cast(Raster, op_tester.run(raster=raster)["index"])
    except ValueError as e:
        if not should_fail:
            raise ValueError(f"this should not have failed. {e}") from e
        return
    output_array = rio.open_rasterio(output.raster_asset.path_or_url).values  # type: ignore
    true_array = true_index_fn[index](da).values
    assert np.all(np.isclose(output_array, true_array))  # type: ignore
