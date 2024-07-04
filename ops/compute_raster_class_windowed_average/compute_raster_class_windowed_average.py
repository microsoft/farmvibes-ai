import logging
from tempfile import TemporaryDirectory
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from vibe_core.data import Raster, gen_guid
from vibe_lib.raster import (
    RGBA,
    interpolated_cmap_from_colors,
    json_to_asset,
    load_raster,
    load_raster_match,
    save_raster_to_asset,
)

CMAP_INTERVALS: List[float] = [0.0, 4000.0]

CMAP_COLORS: List[RGBA] = [
    RGBA(0, 0, 0, 255),
    RGBA(255, 255, 255, 255),
]

LOGGER = logging.getLogger(__name__)


def run_average_elevation(
    dem: NDArray[Any], cdl: NDArray[Any], window_size: int = 41
) -> NDArray[Any]:
    kernel = torch.ones((1, 1, window_size, window_size))
    padding = (window_size - 1) // 2
    eps = 1e-9

    dem_torch = torch.from_numpy(dem).to(kernel)
    cdl_torch = torch.from_numpy(cdl).to(kernel)

    # Downscale
    downscale = 4
    dem_torch = F.interpolate(
        dem_torch.unsqueeze(0),
        (dem_torch.shape[1] // downscale, dem_torch.shape[2] // downscale),
        mode="bilinear",
    ).squeeze(0)

    cdl_torch = F.interpolate(
        cdl_torch.unsqueeze(0),
        (cdl_torch.shape[1] // downscale, cdl_torch.shape[2] // downscale),
        mode="nearest",
    ).squeeze(0)

    # DEM z-scores
    cdl_elevation = torch.zeros_like(dem_torch).to(kernel)

    mean_elev = F.conv2d(
        F.pad(
            dem_torch.unsqueeze(0).to(kernel),
            (padding, padding, padding, padding),
            mode="replicate",
        ),
        kernel,
        bias=None,
        stride=1,
        padding=0,
    ).squeeze(0) / (window_size**2)

    std_elev = F.conv2d(
        F.pad(
            (dem_torch - mean_elev).unsqueeze(0).to(kernel) ** 2,
            (padding, padding, padding, padding),
            mode="replicate",
        ),
        kernel,
        bias=None,
        stride=1,
        padding=0,
    ).squeeze(0) / (window_size**2 - 1)

    # Compute Z-scores of per-class means (wrt statistics of the whole window)
    z_elevation = (dem_torch - mean_elev) / (std_elev + eps)

    # Compute elevation mean per-class in overlapping windows
    unique_cdl_labels = torch.unique(cdl_torch)
    for i in unique_cdl_labels:
        label_mask = cdl_torch == i
        masked_elev = z_elevation * label_mask
        elev_sum = F.conv2d(
            masked_elev.unsqueeze(0), kernel, bias=None, stride=1, padding=padding
        ).squeeze(0)
        label_count = F.conv2d(
            label_mask.unsqueeze(0).to(kernel), kernel, bias=None, stride=1, padding=padding
        ).squeeze(0)
        cdl_elevation[label_mask] = elev_sum[label_mask] / label_count[label_mask]

    # Upsample to original resolution
    cdl_elevation = F.interpolate(
        cdl_elevation.unsqueeze(0), (dem.shape[1], dem.shape[2]), mode="bilinear"
    ).squeeze(0)

    return cdl_elevation.numpy()


class CallbackBuilder:
    def __init__(
        self,
        window_size: int,
    ):
        self.tmp_dir = TemporaryDirectory()
        self.window_size = window_size

    def __call__(self):
        def operator_callback(
            input_dem_raster: Raster, input_cluster_raster: Raster
        ) -> Dict[str, Raster]:
            dem_da = load_raster_match(
                input_dem_raster,
                match_raster=input_cluster_raster,
            )
            cluster_da = load_raster(input_cluster_raster, use_geometry=True)

            average_elevation_da: NDArray[Any] = run_average_elevation(
                dem_da.to_numpy(), cluster_da.to_numpy(), self.window_size
            )

            vis_dict: Dict[str, Any] = {
                "bands": [0],
                "colormap": interpolated_cmap_from_colors(CMAP_COLORS, CMAP_INTERVALS),
                "range": (0, 4000),
            }

            asset = save_raster_to_asset(
                dem_da[:1].copy(data=average_elevation_da), self.tmp_dir.name
            )
            out_raster = Raster.clone_from(
                src=input_dem_raster,
                id=gen_guid(),
                assets=[
                    asset,
                    json_to_asset(vis_dict, self.tmp_dir.name),
                ],
            )

            return {"output_raster": out_raster}

        return operator_callback

    def __del__(self):
        self.tmp_dir.cleanup()
