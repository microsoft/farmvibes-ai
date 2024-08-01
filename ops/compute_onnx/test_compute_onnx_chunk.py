# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, cast

import numpy as np
import pytest
import rioxarray
import torch
import xarray as xr
from numpy.typing import NDArray
from shapely import geometry as shpg
from torch import nn

from vibe_core.data import DataVibe, Raster, RasterChunk, RasterSequence
from vibe_core.data.core_types import gen_guid
from vibe_dev.testing.op_tester import OpTester
from vibe_lib.raster import save_raster_to_asset

N_SAMPLES = 100
STEP_Y = 3
STEP_X = 3
WINDOW_SIZE = 3

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNK_RASTER_YAML = os.path.join(HERE, "..", "chunk_raster", "chunk_raster.yaml")
LIST_TO_SEQ_YAML = os.path.join(HERE, "..", "list_to_sequence", "list_to_sequence.yaml")
COMPUTE_ONNX_YAML = os.path.join(HERE, "compute_onnx_from_chunks.yaml")
COMBINE_CHUNKS_YAML = os.path.join(HERE, "..", "combine_chunks", "combine_chunks.yaml")


class TestModel(nn.Module):
    __test__ = False

    def __init__(self, n: int):
        super(TestModel, self).__init__()
        self.n = n
        A = np.stack((np.arange(n), np.ones(n))).T
        self.A = torch.from_numpy(A)
        self.ATAinv = torch.from_numpy(np.linalg.inv(A.T @ A))

    def forward(self, x: torch.Tensor):
        x = torch.squeeze(x)
        B = torch.reshape(x, (self.n, -1))
        ATB = torch.sum(self.A.reshape(self.n, 2, 1) * B.reshape(self.n, 1, -1), dim=0)
        beta_hat = (self.ATAinv @ ATB)[0, :]
        alpha = (self.ATAinv @ ATB)[1, :]
        return torch.stack((beta_hat.reshape(x.shape[1:]), alpha.reshape(x.shape[1:])))[
            None, :, :, :
        ]


def create_list_fake_raster(
    tmp_dir_name: str, t: int, y: int, x: int
) -> Tuple[List[Raster], NDArray[np.float32]]:
    def fake_cube(sx: int, sy: int, sz: int):
        res = []
        for i in range(sy * sx):
            h = i / (sy * sx - 1)
            res.append(np.linspace(0, h * (sz - 1), sz))

        res = np.stack(res)
        return res.reshape((sy, sx, -1)).transpose((2, 0, 1))

    sf = fake_cube(x, y, t)
    res = []
    for i in range(t):
        fake_da = xr.DataArray(
            sf[i : i + 1, :, :],
            coords={"bands": [0], "x": np.linspace(0, 1, x), "y": np.linspace(0, 1, y)},
            dims=["bands", "y", "x"],
        )
        asset = save_raster_to_asset(fake_da, tmp_dir_name)

        d = datetime(2022, 1, 1) + timedelta(days=i)
        res.append(
            Raster(
                id=gen_guid(),
                time_range=(d, d),
                geometry=shpg.mapping(shpg.box(*fake_da.rio.bounds())),
                bands={"band": 0},
                assets=[asset],
            )
        )

    return res, sf


@pytest.mark.parametrize("y, x", [(6, 6), (3, 3), (6, 3), (3, 6), (8, 3), (8, 8), (10, 12)])
def test_op(y: int, x: int, tmp_path: Path):
    raster_list, input_model = create_list_fake_raster(str(tmp_path.absolute()), N_SAMPLES, y, x)
    model = TestModel(N_SAMPLES)
    model_path = os.path.join(str(tmp_path.absolute()), "model.onnx")
    dummy = np.random.random((1, N_SAMPLES, STEP_Y, STEP_X)).astype(np.float32)
    torch.onnx.export(
        model,
        torch.from_numpy(dummy),
        model_path,
        input_names=["in"],
        output_names=["out"],
        dynamic_axes={"in": {0: "batch", 2: "y", 3: "x"}, "out": {0: "batch", 2: "y", 3: "x"}},
    )

    chunk_raster_op = OpTester(CHUNK_RASTER_YAML)
    chunk_raster_op.update_parameters({"step_y": STEP_Y, "step_x": STEP_X})
    chunked_rasters = cast(
        List[RasterChunk],
        # pyright misidentifies types here
        chunk_raster_op.run(rasters=cast(List[DataVibe], raster_list))[  # type: ignore
            "chunk_series"
        ],
    )

    list_to_raster_op = OpTester(LIST_TO_SEQ_YAML)
    raster_seq = cast(
        RasterSequence,
        # pyright misidentifies types here
        list_to_raster_op.run(list_rasters=cast(List[DataVibe], raster_list))[  # type: ignore
            "rasters_seq"
        ],
    )

    out_chunks = []
    ops = []
    for chunk in chunked_rasters:
        compute_onnx_op = OpTester(COMPUTE_ONNX_YAML)
        compute_onnx_op.update_parameters(
            {
                "root_dir": HERE,
                "model_file": model_path,
                "window_size": WINDOW_SIZE,
                "downsampling": 1,
                "overlap": 0,
            }
        )
        ops.append(compute_onnx_op)
        out_chunks.append(
            cast(
                RasterChunk,
                compute_onnx_op.run(input_raster=cast(DataVibe, raster_seq), chunk=chunk)[
                    "output_raster"
                ],
            )
        )

    combine_chunks_op = OpTester(COMBINE_CHUNKS_YAML)
    output_data = cast(Raster, combine_chunks_op.run(chunks=out_chunks)["raster"])
    output_array = np.squeeze(
        rioxarray.open_rasterio(output_data.raster_asset.path_or_url).values  # type: ignore
    )

    pred_torch = model.forward(torch.from_numpy(input_model[None, :, :, :].astype(np.float32)))
    pred = np.squeeze(pred_torch.detach().numpy())

    assert np.all(np.isclose(output_array, pred))
