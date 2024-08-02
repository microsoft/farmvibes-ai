# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from datetime import datetime
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rioxarray as rio
import xarray as xr
from shapely import geometry as shpg

from vibe_core.data import AssetVibe, GeometryCollection
from vibe_core.data.core_types import gen_guid
from vibe_core.data.rasters import CategoricalRaster, Raster, SamMaskRaster
from vibe_core.data.sentinel import Sentinel2Raster
from vibe_dev.testing.op_tester import OpTester
from vibe_lib.raster import save_raster_to_asset

CONFIG_PATH_PROMPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "prompt_segmentation.yaml"
)

CONFIG_PATH_AUTOSEG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "automatic_segmentation.yaml"
)

DEFAULT_AUTOSEG_PARAMETERS = {
    "points_per_side": 2,
    "spatial_overlap": 0.0,
    "pred_iou_thresh": 0.88,
    "stability_score_thresh": 0.95,
    "n_crop_layers": 0,
    "crop_overlap_ratio": 0.0,
    "crop_n_points_downscale_factor": 1,
}

# Minimum threshold just to make sure the threshold won't remove any masks
MIN_THRESHOLD = 0.00001
BAND_NAMES = {"s2": ["R", "G", "B"], "basemap": ["red", "green", "blue"]}


def edit_autoseg_parameters(key: str, value: Union[int, float]) -> Dict[str, Union[int, float]]:
    new_params = DEFAULT_AUTOSEG_PARAMETERS.copy()
    new_params[key] = value
    return new_params


def create_base_raster(
    tmp_dir_name: str,
    raster_size: int = 2048,
    type: str = "s2",
    cells_per_side: int = 2,
) -> Union[Sentinel2Raster, Raster]:
    now = datetime.now()
    geom = shpg.mapping(shpg.box(0, 0, raster_size, raster_size))

    n_channels = 12 if type == "s2" else 3
    raster_dim = (n_channels, raster_size, raster_size)  # enough for two chips/side

    # Create a checkboard pattern
    cell_size = raster_size // cells_per_side
    row, col = np.indices((raster_size, raster_size))
    pattern_2d = (row // cell_size % 2) ^ (col // cell_size % 2)
    fake_data = 10000.0 * np.repeat(pattern_2d[np.newaxis, :, :], n_channels, axis=0)

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

    if type == "s2":
        raster = Sentinel2Raster(
            id="s2",
            time_range=(now, now),
            geometry=geom,
            assets=[asset],
            bands={
                **{b: idx for idx, b in enumerate(BAND_NAMES[type])},
                **{str(idx): idx for idx in range(3, raster_dim[0])},
            },
            tile_id="",
            processing_level="",
            product_name="",
            orbit_number=0,
            relative_orbit_number=0,
            orbit_direction="",
            platform="",
            extra_info={},
        )
    else:
        raster = Raster(
            id="basemap",
            time_range=(now, now),
            geometry=geom,
            assets=[asset],
            bands={
                **{b: idx for idx, b in enumerate(BAND_NAMES[type])},
                **{str(idx): idx for idx in range(3, raster_dim[0])},
            },
        )

    return raster


def create_geometry_collection(
    prompt_list: List[Union[shpg.Point, shpg.Polygon]],
    label: List[int],
    prompt_id: List[int],
    geom: Dict[str, Any],
    time_range: Tuple[datetime, datetime],
    tmp_dir_name: str,
    column_names: List[str] = ["geometry", "label", "prompt_id"],
):
    df = pd.DataFrame(
        {col_name: info for col_name, info in zip(column_names, [prompt_list, label, prompt_id])}
    )
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")  # type: ignore
    path = os.path.join(tmp_dir_name, "fake_gdf.geojson")
    gdf.to_file(path, driver="GeoJSON")

    asset = AssetVibe(reference=path, type="application/json", id=gen_guid())
    geom_collection = GeometryCollection(
        id=gen_guid(), geometry=geom, time_range=time_range, assets=[asset]
    )
    return geom_collection


@pytest.fixture
def tmp_dir():
    _tmp_dir = TemporaryDirectory()
    yield _tmp_dir.name
    _tmp_dir.cleanup()


@pytest.mark.parametrize(
    "prompt_list, label, prompt_id, expected_exception",
    [
        (
            [shpg.MultiPoint([[1, 1], [2, 2]])],
            [1],
            [0],
            "Expected each geometry to be a shapely Point or Polygon",
        ),
        (
            [shpg.Point(4000, 4000)],  # outside of the raster
            [1],
            [0],
            "Expected all prompts to be contained within the ROI of input_geometry",
        ),
        ([shpg.Point(1, 1)], [1], [5.5], "Expected prompt_ids as integers or strings"),
        (
            [shpg.Point(1, 1), shpg.Point(2, 2)],
            ["a", 5.5],
            [0, 1],
            "Expected labels to be integers, with 0 or 1 values",
        ),
        (
            [shpg.box(1, 1, 2, 2), shpg.box(2, 2, 3, 3)],
            [1, 1],
            [0, 0],
            "Expected at most one bounding box per prompt",
        ),
    ],
)
def test_invalid_prompt_format(
    prompt_list: List[Union[shpg.Point, shpg.Polygon]],
    label: List[int],
    prompt_id: List[int],
    expected_exception: Optional[str],
    tmp_dir: str,
):
    raster = create_base_raster(tmp_dir)
    geom_collection = create_geometry_collection(
        prompt_list=prompt_list,
        label=label,
        prompt_id=prompt_id,
        geom=raster.geometry,
        time_range=raster.time_range,
        tmp_dir_name=tmp_dir,
    )

    with pytest.raises(ValueError, match=expected_exception):
        OpTester(CONFIG_PATH_PROMPT).run(input_raster=raster, input_prompts=geom_collection)


def test_invalid_geometry_collection(tmp_dir: str):
    raster = create_base_raster(tmp_dir)
    geom_collection = create_geometry_collection(
        prompt_list=[shpg.Point(5, 5)],
        label=[1],
        prompt_id=[0],
        geom=raster.geometry,
        time_range=raster.time_range,
        tmp_dir_name=tmp_dir,
        column_names=["geometry", "label", "wrong_column_name"],
    )

    with pytest.raises(ValueError):
        OpTester(CONFIG_PATH_PROMPT).run(input_raster=raster, input_prompts=geom_collection)


# Points expressed as fraction of the raster size for easier conversion to pixel coordinates
@pytest.mark.parametrize(
    "raster_type, raster_size, spatial_overlap, prompt_list, label, prompt_id, expected_mask_area",
    [
        (  # One point per quadrant as separate prompts
            "s2",
            2048,
            0.0,
            [
                shpg.Point(0.25, 0.25),  # top-left quadrant
                shpg.Point(0.75, 0.25),  # top-right quadrant
                shpg.Point(0.25, 0.75),  # bottom-left quadrant
                shpg.Point(0.75, 0.75),  # bottom-right quadrant
            ],
            [1, 1, 1, 1],
            [0, 1, 2, 3],
            1024 * 1024,  # one quadrant, 1/4 of the raster area
        ),
        (  # One prompt with 2 points on the top-left and bottom-right quadrants
            "basemap",
            2048,
            0.0,
            [
                shpg.Point(0.25, 0.25),
                shpg.Point(0.75, 0.25),
                shpg.Point(0.25, 0.75),
                shpg.Point(0.75, 0.75),
            ],
            [1, 0, 0, 1],
            [0, 0, 0, 0],
            2 * 1024 * 1024,  # two quadrant, 1/2 of the raster area
        ),
        (  # Four points per quadrant, each quadrant as separate prompt
            "s2",
            2048,
            0.0,
            [shpg.Point(0.125 + i * 0.25, 0.125 + j * 0.25) for i in range(4) for j in range(4)],
            [1] * 16,
            [2 * (i // 2) + (j // 2) for i in range(4) for j in range(4)],
            1024 * 1024,  # one quadrant, 1/4 of the raster area
        ),
        (  # Four points per quadrant, single prompt (top-left, bottom-right), 50% of overlap
            "basemap",
            2048,
            0.0,
            [shpg.Point(0.125 + i * 0.25, 0.125 + j * 0.25) for i in range(4) for j in range(4)],
            [1, 1, 0, 0] * 2 + [0, 0, 1, 1] * 2,
            [1] * 16,
            2 * 1024 * 1024,  # two quadrant, 1/2 of the raster area
        ),
        (  # Bbox of half of a quadrant centered in the first quadrant, single prompt, no overlap
            "s2",
            2048,
            0.0,
            [shpg.box(0.125, 0.125, 0.375, 0.375)],
            [1],
            [0],
            512 * 512,  # half quadrant, 1/8 of the raster area
        ),
        (  # Same Bbox as above with a centered foreground point, single prompt, no overlap
            "basemap",
            2048,
            0.0,
            [shpg.box(0.125, 0.125, 0.375, 0.375), shpg.Point(0.25, 0.25)],
            [1, 1],
            [0, 0],
            512 * 512,  # half quadrant, 1/8 of the raster area
        ),
    ],
)
def test_segmentation_mask(
    raster_type: str,
    raster_size: int,
    spatial_overlap: float,
    prompt_list: List[Union[shpg.Point, shpg.Polygon]],
    label: List[int],
    prompt_id: List[int],
    expected_mask_area: int,
    tmp_dir: str,
):
    raster = create_base_raster(tmp_dir, raster_size, raster_type)
    geom_collection = create_geometry_collection(
        prompt_list=prompt_list,
        label=label,
        prompt_id=prompt_id,
        geom=raster.geometry,
        time_range=raster.time_range,
        tmp_dir_name=tmp_dir,
    )

    op_tester = OpTester(CONFIG_PATH_PROMPT)
    op_tester.update_parameters(
        {"spatial_overlap": spatial_overlap, "band_names": BAND_NAMES[raster_type]}
    )
    output = op_tester.run(input_raster=raster, input_prompts=geom_collection)

    assert "segmentation_mask" in output

    mask_raster = cast(CategoricalRaster, output["segmentation_mask"])
    assert len(mask_raster.bands) == len(np.unique(prompt_id))

    mask = rio.open_rasterio(mask_raster.assets[0].path_or_url).values  # type: ignore
    assert mask.shape == (len(np.unique(prompt_id)), 2048, 2048)

    for idx, _ in enumerate(np.unique(prompt_id)):
        assert (
            np.abs(
                np.sum(mask[idx, :, :]) - expected_mask_area  # type: ignore
            )
            <= 0.05 * expected_mask_area
        ), "Mask area is not within 5 percent of the expected area"


@pytest.mark.parametrize(
    "raster_type, checkboard_cells_per_side, points_per_side, spatial_overlap, "
    "pred_iou_thresh, stability_score_thresh, n_crop_layers, n_expected_masks",
    [
        (
            "s2",
            2,
            2,
            DEFAULT_AUTOSEG_PARAMETERS["spatial_overlap"],
            MIN_THRESHOLD,
            MIN_THRESHOLD,
            DEFAULT_AUTOSEG_PARAMETERS["n_crop_layers"],
            16,
        ),  # 2x2 raster, 4 chips, 4 masks/chip (2pps**2) = 16 masks
        (
            "basemap",
            2,
            2,
            DEFAULT_AUTOSEG_PARAMETERS["spatial_overlap"],
            MIN_THRESHOLD,
            MIN_THRESHOLD,
            1,
            80,  # 16 masks for crop layer 0 + 4*16 for the next layer
        ),  # Same as above, but with an additional crop layer
        (
            "s2",
            2,
            2,
            0.5,
            MIN_THRESHOLD,
            MIN_THRESHOLD,
            DEFAULT_AUTOSEG_PARAMETERS["n_crop_layers"],
            36,  # SAM removes a few due to low quality and stability scores
        ),  # 2x2 raster, 9 chips (due to overlap), 4 masks/chip (2pps**2) = 36 masks
        (
            "basemap",
            2,
            2,
            0.5,
            DEFAULT_AUTOSEG_PARAMETERS["pred_iou_thresh"],
            DEFAULT_AUTOSEG_PARAMETERS["stability_score_thresh"],
            DEFAULT_AUTOSEG_PARAMETERS["n_crop_layers"],
            31,  # SAM removes a few due to low quality and stability scores
        ),  # Same as above, but with filtered masks
        (
            "s2",
            4,
            4,
            DEFAULT_AUTOSEG_PARAMETERS["spatial_overlap"],
            MIN_THRESHOLD,
            MIN_THRESHOLD,
            DEFAULT_AUTOSEG_PARAMETERS["n_crop_layers"],
            64,  # Without the IoU quality and stability score filtering, we expect all 64 masks
        ),  # 4x4 raster, 4 chips, 16 masks/chip (4pps**2) = 64 masks
        (
            "basemap",
            4,
            4,
            DEFAULT_AUTOSEG_PARAMETERS["spatial_overlap"],
            DEFAULT_AUTOSEG_PARAMETERS["pred_iou_thresh"],
            DEFAULT_AUTOSEG_PARAMETERS["stability_score_thresh"],
            DEFAULT_AUTOSEG_PARAMETERS["n_crop_layers"],
            36,  # SAM removes a few due to low quality and stability scores
        ),  # Same as above, but with filtered masks
    ],
)
def test_automatic_segmentation_mask(
    raster_type: str,
    checkboard_cells_per_side: int,
    points_per_side: int,
    spatial_overlap: float,
    pred_iou_thresh: float,
    stability_score_thresh: float,
    n_crop_layers: int,
    n_expected_masks: int,
    tmp_dir: str,
):
    raster_size = 2048
    raster = create_base_raster(tmp_dir, raster_size, raster_type, checkboard_cells_per_side)

    op_tester = OpTester(CONFIG_PATH_AUTOSEG)
    op_tester.update_parameters(
        {
            "points_per_side": points_per_side,
            "spatial_overlap": spatial_overlap,
            "n_crop_layers": n_crop_layers,
            "pred_iou_thresh": pred_iou_thresh,
            "stability_score_thresh": stability_score_thresh,
            "band_names": BAND_NAMES[raster_type],
        }
    )
    output = op_tester.run(input_raster=raster)

    assert "segmented_chips" in output

    segmented_chips = cast(List[SamMaskRaster], output["segmented_chips"])
    step_size = 1024 * (1 - spatial_overlap)
    n_expected_rasters = (1 + (raster_size - 1024) / step_size) ** 2
    assert len(segmented_chips) == n_expected_rasters, (
        "Unexpected number of output rasters. "
        f"Got {len(segmented_chips)}, expected {n_expected_rasters}."
    )

    n_masks = 0
    mask_areas = []
    for chip in segmented_chips:
        mask = cast(xr.Dataset, rio.open_rasterio(chip.assets[0].path_or_url)).values
        mask_areas.extend(np.sum(mask, axis=(1, 2)).reshape(-1).tolist())  # type: ignore
        n_masks += mask.shape[0]

    assert (
        n_masks == n_expected_masks
    ), f"Unexpected number of output masks. Got {n_masks}, expected {n_expected_masks}."


@pytest.mark.parametrize(
    "param_key, invalid_value",
    [
        ("points_per_side", 0),
        ("points_per_side", 1.5),
        ("n_crop_layers", -1),
        ("n_crop_layers", 1.5),
        ("crop_overlap_ratio", -1),
        ("crop_overlap_ratio", 1.5),
        ("crop_n_points_downscale_factor", 0),
        ("crop_n_points_downscale_factor", 1.5),
        ("pred_iou_thresh", 0),
        ("pred_iou_thresh", 1),
        ("stability_score_thresh", 0),
        ("stability_score_thresh", 1.5),
        ("band_names", ["Cyan", "Magenta", "Yellow"]),
        ("band_names", ["R", "G", "B", "N", "N2"]),
        ("band_names", ["R", "G"]),
        ("band_scaling", [1.0, 1.0]),
        ("band_offset", [1.0, 1.0]),
    ],
)
def test_invalid_autoseg_params(
    param_key: str,
    invalid_value: Union[int, float],
    tmp_dir: str,
):
    raster = create_base_raster(tmp_dir)
    op_tester = OpTester(CONFIG_PATH_AUTOSEG)
    op_tester.update_parameters(edit_autoseg_parameters(param_key, invalid_value))
    with pytest.raises(ValueError):
        op_tester.run(input_raster=raster)
