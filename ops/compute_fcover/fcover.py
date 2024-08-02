# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
FCOVER computation using a neural network as described in
https://step.esa.int/docs/extra/ATBD_S2ToolBox_L2B_V1.1.pdf
https://github.com/senbox-org/s2tbx/blob/master/s2tbx-biophysical/src/main/java/org/esa/s2tbx/biophysical
https://www.sciencedirect.com/science/article/pii/S0034425710002853
https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/fcover/

Following implementation from Sentinel-2 Toolbox
https://github.com/senbox-org/s2tbx/blob/master/s2tbx-biophysical/src/main/java/org/esa/s2tbx/biophysical/BiophysicalOp.java

Normalization params and weights from Sentinel-2 Toolbox for L2A
https://github.com/senbox-org/s2tbx/tree/master/s2tbx-biophysical/src/main/resources/auxdata/3_0/S2A/FCOVER
"""

from tempfile import TemporaryDirectory
from typing import Any, Dict, cast, overload

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from rasterio.warp import Resampling

from vibe_core.data import Raster, gen_guid
from vibe_lib.raster import get_cmap, json_to_asset, load_raster, save_raster_to_asset

BANDS = ["B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12"]

# Normalization params: min - max for each band
BANDS_NORM = np.array(
    (
        (0, 0.23901527463861838),
        (0, 0.29172736471507876),
        (0, 0.32652671459255694),
        (0.008717364330310326, 0.5938903910368211),
        (0.019693160430621366, 0.7466909927207045),
        (0.026217828282102625, 0.7582393779705984),
        (0.018931934894415213, 0.4929337190581187),
        (0, 0.4877499217101771),
    )
)
ANGLES_NORM = np.array(
    (
        (0.979624800125421, 0.9999999999691099),
        (0.342108564072183, 0.9274847491748729),
        (-0.9999999986740542, 0.9999999998869543),
    )
)

DENORMALIZATION = np.array((0.0001143371095669865, 0.9994883064311412))

# NN Weights
# Layer 1: 5 hidden neurons
# 5 x 11 matrix
W1 = np.array(
    (
        (
            -0.09299549787532572,
            0.03711751310275837,
            0.35917948087916934,
            -2.0327599053936245,
            -0.3004739931440174,
            5.081364269387806,
            -0.5509229514856009,
            -1.8459014400791363,
            0.04210879716286216,
            -0.1433820536680042,
            -0.0919637992244123,
        ),
        (
            0.17782538722557306,
            -0.3793824396587722,
            -0.18316058499587165,
            -0.8546862528226032,
            -0.07553090207841909,
            2.1968612305059834,
            -0.1734580018542482,
            -0.89158072360678,
            0.017977829778812265,
            0.19161704265110313,
            -0.020341567456493917,
        ),
        (
            -0.8964833683739212,
            -0.6038768961220443,
            -0.5995953059405849,
            -0.15212446911598965,
            0.3889544003539062,
            1.9871015442471918,
            -0.9746781245763875,
            -0.28459612830995773,
            -0.7195016395928718,
            0.4628341672035696,
            1.652035259226453,
        ),
        (
            -0.15296262636768043,
            0.17628558201043018,
            0.11212126329600514,
            1.5711153194443364,
            0.5209619736717268,
            -3.068192837466073,
            0.1483332044127799,
            1.2331177561153577,
            -0.02091226761957991,
            -0.23041694611129848,
            0.0031568086031440803,
        ),
        (
            1.7234228895153363,
            -2.906528582039084,
            -1.3938598383149996,
            -1.6262956756929428,
            0.3326361580291295,
            -0.8862583674506147,
            -0.2185426118098439,
            0.5660635905206617,
            -0.09949171171933309,
            -0.35271418843339297,
            0.06514559686105968,
        ),
    )
)
B1 = np.array(
    (
        -1.886007283361096,
        -0.02498619641898423,
        0.29510485628465327,
        0.0029300996499639458,
        -3.359449911074414,
    )
)
# Layer 2: 1 output neuron
# 1 x 5 matrix
W2 = np.array(
    (
        0.21418510066217855,
        2.354410480678047,
        0.039929632100371135,
        1.5480571230482811,
        -0.11310020940549115,
    )
)

B2 = -0.15076057408085747


def fcover_fun(raster: xr.DataArray, angles: xr.DataArray) -> xr.DataArray:
    # Normalize bands
    norm_bands = normalize(raster, BANDS_NORM[:, :1, None], BANDS_NORM[:, 1:, None])
    # Normalize angles before upsampling
    zen_norm = normalize(
        cast(xr.DataArray, np.cos(np.deg2rad(angles[[0, 2]]))),
        ANGLES_NORM[:2, :1, None],
        ANGLES_NORM[:2, 1:, None],
    )
    rel_az_norm = cast(
        xr.DataArray,
        normalize(
            np.cos(np.deg2rad(angles[3] - angles[1])),
            ANGLES_NORM[2, :1, None],
            ANGLES_NORM[2, 1:, None],
        ),
    ).expand_dims("band")
    norm_angles = xr.concat((zen_norm, rel_az_norm), dim="band")
    # Upsample angles to the same resolution as the band data
    norm_angles = norm_angles.rio.reproject_match(norm_bands, resampling=Resampling.bilinear)
    full_data = xr.concat((norm_bands, norm_angles), dim="band").to_numpy()
    layer1 = np.tanh(W1.dot(full_data.transpose((1, 0, 2))) + B1[:, None, None])
    layer2 = np.tanh(W2.dot(layer1.transpose(1, 0, 2)) + B2)
    fcover = denormalize(layer2, DENORMALIZATION[0], DENORMALIZATION[1])[None]
    fcover = raster[:1].copy(data=fcover)  # Copy metadata
    return fcover


@overload
def normalize(unnormalized: NDArray[Any], min: NDArray[Any], max: NDArray[Any]) -> NDArray[Any]: ...


@overload
def normalize(unnormalized: xr.DataArray, min: NDArray[Any], max: NDArray[Any]) -> xr.DataArray: ...


def normalize(unnormalized: Any, min: NDArray[Any], max: NDArray[Any]):
    return 2 * (unnormalized - min) / (np.subtract(max, min)) - 1


@overload
def denormalize(normalized: NDArray[Any], min: NDArray[Any], max: NDArray[Any]) -> NDArray[Any]: ...


@overload
def denormalize(normalized: xr.DataArray, min: NDArray[Any], max: NDArray[Any]) -> xr.DataArray: ...


def denormalize(normalized: Any, min: NDArray[Any], max: NDArray[Any]):
    return 0.5 * (normalized + 1) * (np.subtract(max, min)) + min


class CallbackBuilder:
    def __init__(self):
        self.tmp_dir = TemporaryDirectory()

    def __call__(self):
        def fcover_callback(raster: Raster, angles: Raster) -> Dict[str, Raster]:
            r = load_raster(raster, bands=BANDS, use_geometry=True) * raster.scale + raster.offset
            a = load_raster(angles, use_geometry=True)
            fcover = fcover_fun(r, a)
            asset = save_raster_to_asset(fcover, self.tmp_dir.name)
            vis_dict = {
                "bands": [0],
                "colormap": get_cmap("viridis"),
                "range": (0, 1),
            }
            out_raster = Raster.clone_from(
                raster,
                id=gen_guid(),
                assets=[asset, json_to_asset(vis_dict, self.tmp_dir.name)],
                bands={"fcover": 0},
            )
            return {"fcover": out_raster}

        return fcover_callback

    def __del__(self):
        self.tmp_dir.cleanup()
