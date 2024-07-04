import os
from datetime import datetime
from tempfile import TemporaryDirectory
from typing import List, Tuple, Union, cast

import numpy as np
import pytest
import xarray as xr
from shapely import geometry as shpg

from vibe_core.data.core_types import ChipWindow, gen_guid
from vibe_core.data.rasters import CategoricalRaster, SamMaskRaster
from vibe_dev.testing.op_tester import OpTester
from vibe_lib.raster import save_raster_to_asset

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "combine_sam_masks.yaml")

DEFAULT_BBOXES = [
    (0, 0, 1024, 1024),
    (1024, 0, 2048, 1024),
    (0, 1024, 1024, 2048),
    (1024, 1024, 2048, 2048),
]


def create_segmented_raster(
    tmp_dir_name: str,
    mask_bbox: Tuple[int, int, int, int],
    mask_score: float = 1.0,
    raster_size: int = 2048,
) -> SamMaskRaster:
    now = datetime.now()
    geom = shpg.mapping(shpg.box(0, 0, raster_size, raster_size))

    raster_dim = (1, raster_size, raster_size)

    fake_data = np.zeros(raster_dim, dtype=np.uint8)
    fake_data[0, mask_bbox[1] : mask_bbox[3], mask_bbox[0] : mask_bbox[2]] = 1

    fake_da = xr.DataArray(
        fake_data,
        coords={
            "bands": np.arange(raster_dim[0]),
            "x": np.linspace(0, 1, raster_dim[1]),
            "y": np.linspace(0, 1, raster_dim[2]),
        },
        dims=["bands", "y", "x"],
    )
    fake_da.rio.write_crs("epsg:4326", inplace=True)

    asset = save_raster_to_asset(fake_da, tmp_dir_name)

    return SamMaskRaster(
        id=gen_guid(),
        time_range=(now, now),
        geometry=geom,
        assets=[asset],
        bands={"mask": 0},
        categories=["background", "foreground"],
        mask_score=[mask_score],
        mask_bbox=[tuple([float(c) for c in mask_bbox])],  # type: ignore
        chip_window=ChipWindow(0.0, 0.0, float(raster_size), float(raster_size)),
    )


@pytest.fixture
def tmp_dir():
    _tmp_dir = TemporaryDirectory()
    yield _tmp_dir.name
    _tmp_dir.cleanup()


@pytest.mark.parametrize(
    "param_key, invalid_value",
    [(p, v) for p in ["chip_nms_thr", "mask_nms_thr"] for v in [-1, 0, 1, 1.5]],
)
def test_invalid_params(
    param_key: str,
    invalid_value: Union[int, float],
    tmp_dir: str,
):
    raster = create_segmented_raster(tmp_dir, mask_bbox=(0, 0, 1024, 1024))
    op_tester = OpTester(CONFIG_PATH)
    op_tester.update_parameters({param_key: invalid_value})
    with pytest.raises(ValueError):
        op_tester.run(input_masks=[raster])


# Points expressed as fraction of the raster size for easier conversion to pixel coordinates
@pytest.mark.parametrize(
    "bbox_list, chip_nms_thr, mask_nms_thr, n_expected_masks",
    [
        (
            DEFAULT_BBOXES,
            0.7,
            0.5,
            4,  # No overlapping masks, so expect the same number
        ),
        (
            DEFAULT_BBOXES + [(10, 10, 1014, 1014)],
            0.7,
            0.5,
            4,  # One mask is completely contained in another
        ),
        (  # Overlapping with top two masks, but with an area slightly larger than a chip
            DEFAULT_BBOXES + [(500, 0, 1550, 1024)],
            0.7,
            0.5,  # threshold of 0.5 IoU won't suppress the new box
            5,  # Overlapping with two masks, but IoU won't pass the threshold so we will keep it
        ),
        (  # Overlapping with top two masks, but with an area slightly larger than a chip
            DEFAULT_BBOXES + [(500, 0, 1550, 1024)],
            0.7,
            0.3,  # lowering the threshold so it will be suppressed (we prefer smaller masks)
            4,
        ),
    ],
)
def test_segmentation_mask(
    bbox_list: List[Tuple[int, int, int, int]],
    chip_nms_thr: float,
    mask_nms_thr: float,
    n_expected_masks: int,
    tmp_dir: str,
):
    input_masks = [create_segmented_raster(tmp_dir, mask_bbox=box) for box in bbox_list]

    op_tester = OpTester(CONFIG_PATH)
    op_tester.update_parameters({"chip_nms_thr": chip_nms_thr, "mask_nms_thr": mask_nms_thr})
    output = op_tester.run(input_masks=input_masks)  # type: ignore

    assert "output_mask" in output

    mask_raster = cast(CategoricalRaster, output["output_mask"])
    assert len(mask_raster.bands) == n_expected_masks
