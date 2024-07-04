import mimetypes
import os
from datetime import datetime, timedelta
from tempfile import TemporaryDirectory
from typing import List, Tuple, Union, cast

import numpy as np
import pytest
import rioxarray
import torch
import xarray as xr
from numpy.typing import NDArray
from shapely import geometry as shpg
from torch import nn
from torch.nn.parameter import Parameter

from vibe_core.data import AssetVibe, Raster
from vibe_core.data.core_types import gen_guid
from vibe_core.data.rasters import RasterSequence
from vibe_dev.testing.op_tester import OpTester

YAML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compute_onnx.yaml")
YAML_FLIST_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "compute_onnx_from_sequence.yaml"
)
PY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compute_onnx.py")


class IdentityNetwork(nn.Module):
    def __init__(self, channels: int):
        super(IdentityNetwork, self).__init__()
        self.c1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, bias=False)
        eye = np.eye(channels).reshape((channels, channels, 1, 1)).astype(np.float32)
        self.c1.weight = Parameter(torch.from_numpy(eye))

    def forward(self, x: torch.Tensor):
        return self.c1(x)


class DummyCloud(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super(DummyCloud, self).__init__()
        self.c1 = nn.Conv2d(
            in_channels=channels,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        w = np.ones((1, channels, kernel_size, kernel_size)).astype(np.float32)
        self.c1.weight = Parameter(torch.from_numpy(w))
        self.p = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        return self.p(self.c1(x))


def create_onnx_model(nn: nn.Module, tmp_dir_name: str, channels: int) -> str:
    dims = (1, channels, 3, 3)  # any value for batch size, y, x should work here
    data = np.random.random(dims).astype(np.float32)

    name = f"{nn.__class__.__name__}.onnx"

    torch.onnx.export(
        nn,
        torch.Tensor(data),
        os.path.join(tmp_dir_name, name),
        input_names=["in"],
        output_names=["out"],
        dynamic_axes={"in": {0: "batch", 2: "y", 3: "x"}, "out": {0: "batch", 2: "y", 3: "x"}},
    )

    return name


def create_fake_raster(
    tmp_dir_name: str, bands: int, y: int, x: int, delta: int = 0
) -> Tuple[Raster, NDArray[np.float32]]:
    fake_data = np.random.random((bands, y, x)).astype(np.float32)
    fake_da = xr.DataArray(
        fake_data,
        coords={"bands": np.arange(bands), "x": np.linspace(0, 1, x), "y": np.linspace(0, 1, y)},
        dims=["bands", "y", "x"],
    )
    path = os.path.join(tmp_dir_name, f"{gen_guid()}.tif")
    fake_da.rio.to_raster(path)

    asset = AssetVibe(
        reference=path,
        type=mimetypes.types_map[".tif"],
        id="fake_asset",
    )

    d = datetime(2022, 1, 1) + timedelta(days=delta)

    return (
        Raster(
            id="fake_id",
            time_range=(d, d),
            geometry=shpg.mapping(shpg.box(*fake_da.rio.bounds())),
            assets=[asset],
            bands={str(i): i for i in range(bands)},
        ),
        fake_data,
    )


@pytest.fixture
def tmp_dir():
    _tmp_dir = TemporaryDirectory()
    yield _tmp_dir.name
    _tmp_dir.cleanup()


@pytest.mark.parametrize(
    "bands, y, x",
    [
        ([3, 2, 1], 512, 512),
        ([2, 2, 2], 1024, 1024),
        ([1], 514, 513),
        (3, 512, 512),
        (2, 1024, 1024),
    ],
)
def test_op(bands: Union[int, List[int]], y: int, x: int, tmp_dir: str):
    model_class_list = [IdentityNetwork, DummyCloud]
    channels = np.sum(bands).astype(int)

    model_list = [m(channels) for m in model_class_list]
    onnx_list = [create_onnx_model(m, tmp_dir, channels) for m in model_list]
    if isinstance(bands, list):
        yaml = YAML_FLIST_PATH
        rasters = []
        arrays = []
        for i, n in enumerate(bands):
            raster, array = create_fake_raster(tmp_dir, n, y, x, delta=i)
            rasters.append(raster)
            arrays.append(array)
        raster = RasterSequence.clone_from(rasters[0], gen_guid(), [])
        for r in rasters:
            raster.add_item(r)
        array = np.concatenate(arrays, axis=0)
    else:
        yaml = YAML_PATH
        raster, array = create_fake_raster(tmp_dir, bands, y, x)

    op_tester = OpTester(yaml)
    for model, onnx in zip(model_list, onnx_list):
        parameters = {"root_dir": tmp_dir, "model_file": onnx, "overlap": 0.1}
        op_tester.update_parameters(parameters)
        output_data = cast(Raster, op_tester.run(input_raster=raster)["output_raster"])
        output_array = rioxarray.open_rasterio(output_data.raster_asset.path_or_url).values  # type: ignore
        true_array = model.forward(torch.from_numpy(array)).detach().numpy()
        assert np.all(np.isclose(output_array, true_array))  # type: ignore
