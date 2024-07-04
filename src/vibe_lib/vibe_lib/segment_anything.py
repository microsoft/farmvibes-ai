import logging
from itertools import product
from math import ceil
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import geopandas as gpd
import numpy as np
import onnxruntime as ort
import shapely.geometry as shpg
import torch
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from rasterio import Affine
from shapely.geometry.base import BaseGeometry
from torchvision.transforms.functional import resize

from vibe_core.data import GeometryCollection, Raster
from vibe_core.data.core_types import BBox, Point
from vibe_lib.spaceeye.chip import ChipDataset, Dims, Window

LOGGER = logging.getLogger(__name__)

SAM_CHIP_SIZE = 1024
SAM_PIXEL_RGB_MEAN = [123.675, 116.28, 103.53]
SAM_PIXEL_RGB_STD = [58.395, 57.12, 57.375]
BACKGROUND_VALUE = 0
MASK_LOGIT_THRESHOLD = 0.0

Prompt = Tuple[Union[Point, BBox], int]


#
# PROMPT VALIDATION and PREPROCESSING METHODS
#


def is_valid_prompt(prompt: List[Prompt], n_original_fg_pnts: int) -> bool:
    """Check if prompt is valid for SAM inference.

    Valid prompts within a chip:
        - Prompt contains at least one foreground point (with or without bbox).
        - Prompt contains bbox without foreground points in the original prompt group/id.

    Args:
        prompt: List of prompts.
        n_original_fg_pnts: Number of original foreground points in the prompt group/id.

    Returns:
        True if prompt is valid, False otherwise.
    """
    if prompt:
        pts_in_chip = [p for p in prompt if len(p[0]) == 2]
        bbox_in_chip = [p for p in prompt if len(p[0]) == 4]
        return (1 in [p[1] for p in pts_in_chip]) or (
            len(bbox_in_chip) > 0 and n_original_fg_pnts == 0
        )
    return False


def validate_prompt_geometry_collection(prompt_df: GeoDataFrame, roi: shpg.Polygon):
    """Validate a GeoDataFrame representing a geometry collection with points or bbox as prompts.

    Args:
        prompt_df: GeoDataFrame with columns 'prompt_id', 'label', and 'geometry'.
        roi: Polygon representing the region of interest.

    Raises:
        ValueError: If prompts are invalid.
    """
    if not all(col in prompt_df.columns for col in ["prompt_id", "label", "geometry"]):
        raise ValueError(
            "Geometry collection must have columns 'prompt_id', 'label', and 'geometry'. "
            f"Columns found: {prompt_df.columns}"
        )

    if not prompt_df.geometry.apply(lambda g: isinstance(g, (shpg.Point, shpg.Polygon))).all():
        prompt_types = list(
            set(
                [
                    type(g)
                    for g in prompt_df.geometry
                    if not (isinstance(g, (shpg.Point, shpg.Polygon)))
                ]
            )
        )
        raise ValueError(
            f"Expected each geometry to be a shapely Point or Polygon. Found: {prompt_types}"
        )

    prompts_within_roi = prompt_df.geometry.within(roi)
    if not prompts_within_roi.all():
        prompts_outside_roi = prompt_df.geometry[~prompts_within_roi]
        coords = [
            (p.x, p.y) if isinstance(p, shpg.Point) else p.bounds for p in prompts_outside_roi
        ]
        raise ValueError(
            "Expected all prompts to be contained within the ROI of input_geometry. Prompts "
            f"outside of ROI: {coords}"
        )

    if not prompt_df.prompt_id.apply(lambda i: isinstance(i, (int, str))).all():
        prompts = [i for i in prompt_df.prompt_id if not isinstance(i, (int, str))]
        raise ValueError(f"Expected prompt_ids as integers or strings. Found: {prompts}")

    if not prompt_df.label.apply(lambda i: isinstance(i, int) and i in (0, 1)).all():
        raise ValueError(
            "Expected labels to be integers, with 0 or 1 values. "
            f"Found: {[i for i in prompt_df.label if not isinstance(i, int) or i not in (0, 1)]}"
        )

    for prompt_id, group in prompt_df.groupby("prompt_id"):
        nbbox = sum([isinstance(g, shpg.Polygon) for g in group.geometry])
        if nbbox > 1:
            raise ValueError(
                "Expected at most one bounding box per prompt. "
                f"Found {nbbox} for prompt_id '{prompt_id}'"
            )


def adjust_bounding_box(prompts: List[Prompt]) -> List[Prompt]:
    """Adjust bounding box coordinates to contain all foreground points in the prompt

    Args:
        prompts: List of prompts.

    Returns:
        Adjusted list of prompts.
    """
    bbox = [p for p in prompts if len(p[0]) == 4]
    foreground_points = [point for point, label in prompts if len(point) == 2 and label == 1]
    if not bbox or not foreground_points:
        return prompts

    bbox_coords, bbox_label = bbox[0]
    xmin, ymin, xmax, ymax = cast(BBox, bbox_coords)

    x_pts, y_pts = zip(*foreground_points)

    xmin, xmax = np.min([xmin, np.min(x_pts)]), np.max([xmax, np.max(x_pts)])
    ymin, ymax = np.min([ymin, np.min(y_pts)]), np.max([ymax, np.max(y_pts)])

    adjusted_prompts = [cast(Prompt, ((xmin, ymin, xmax, ymax), bbox_label))] + [
        p for p in prompts if len(p[0]) == 2
    ]

    return adjusted_prompts


def convert_coords_to_pixel_position(
    geometry: Union[shpg.Point, shpg.Polygon], transform: Affine
) -> Union[Point, BBox]:
    """Convert point/bbox coordinates to pixel position.

    If bounding box, returns the pixel positions as a tuple of (xmin, ymin, xmax, ymax),
    as expected by SAM.

    Args:
        geometry: Point or Polygon geometry.
        transform: Affine transformation matrix.

    Returns:
        Coordinates in pixel position.

    Raises:
        ValueError: If geometry is not a Point or Polygon.
    """

    if isinstance(geometry, shpg.Point):
        return ~transform * (geometry.x, geometry.y)  # type: ignore
    elif isinstance(geometry, shpg.Polygon):
        bounds = geometry.bounds
        pixel_pos = ~transform * bounds[:2] + ~transform * bounds[2:]  # type: ignore
        xmin, xmax = sorted(pixel_pos[::2])
        ymin, ymax = sorted(pixel_pos[1::2])
        return (xmin, ymin, xmax, ymax)
    else:
        raise ValueError(f"Invalid prompt geometry: {geometry}")


def preprocess_geometry_collection(
    geometry_collection: GeometryCollection,
    dataset: ChipDataset,
    roi_geometry: BaseGeometry,
) -> Tuple[Dict[int, List[Prompt]], Dict[int, str]]:
    """Preprocess input geometry collection.

    Args:
        geometry_collection: Geometry collection with prompts.
        dataset: ChipDataset object.
        roi_geometry: Region of interest geometry.
    Returns:
        Tuple of prompts and prompt mapping.
    """
    prompt_df = cast(
        gpd.GeoDataFrame,
        gpd.read_file(geometry_collection.assets[0].path_or_url).to_crs(dataset.meta["crs"]),  # type: ignore
    )
    # Assert GeoDataFrame format and field values
    roi_polygon = cast(
        shpg.Polygon,
        gpd.GeoSeries(roi_geometry, crs="epsg:4326")  # type: ignore
        .to_crs(dataset.crs)
        .iloc[0]
        .envelope,
    )
    try:
        validate_prompt_geometry_collection(prompt_df, roi_polygon)
    except ValueError as e:
        raise ValueError(f"Failed to parse prompts for segmentation. {e}") from e

    # Group by prompt_id and build tuple of transformed points and label pairs
    groups = prompt_df.groupby("prompt_id")
    grouped_prompts = groups.apply(
        lambda x: [
            (convert_coords_to_pixel_position(geometry, dataset.transform), label)
            for geometry, label in zip(x.geometry, x.label)
        ]
    )
    grouped_prompts = cast(Dict[Union[int, str], List[Prompt]], grouped_prompts.to_dict())

    # Adjust bounding box to cover all points within the same prompt
    grouped_prompts = {
        prompt_id: adjust_bounding_box(prompts) for prompt_id, prompts in grouped_prompts.items()
    }

    # Remapping prompt_ids to 0, 1, 2, ...
    prompt_dict = {
        new_id: cast(List[Prompt], grouped_prompts[prompt_id])
        for new_id, prompt_id in enumerate(grouped_prompts.keys())
    }
    prompt_mapping = {
        new_id: str(prompt_id) for new_id, prompt_id in enumerate(grouped_prompts.keys())
    }
    return prompt_dict, prompt_mapping


def get_normalized_prompts_within_chip(
    prompts: Dict[int, List[Prompt]], read_window: Window, geometry_offset: Dims
) -> Dict[int, List[Prompt]]:
    """Filter and normalize prompts within chip.

    Output prompts will include only prompts within the chip with normalized coordinates relative
    to the chip read window.

    Args:
        prompts: Dictionary of prompts.
        read_window: Chip read window.
        geometry_offset: Chip geometry offset.
    Returns:
        Dictionary of normalized prompts.
    """
    col_min = read_window.col_off - geometry_offset.width
    col_max = col_min + read_window.width

    row_min = read_window.row_off - geometry_offset.height
    row_max = row_min + read_window.height

    normalized_prompts = {}
    for prompt_id, prompt in prompts.items():
        new_prompt, n_foreground_points = [], 0
        for coords, lb in prompt:
            if len(coords) == 2:  # Point
                n_foreground_points += lb
                x, y = cast(Point, coords)
                if (col_min <= x <= col_max) and (row_min <= y <= row_max):
                    new_prompt.append(((x - col_min, y - row_min), lb))
            elif len(coords) == 4:  # Bounding box
                xmin, ymin, xmax, ymax = cast(BBox, coords)
                if xmin < col_max and xmax > col_min and ymin < row_max and ymax > row_min:
                    xmin = max(xmin, col_min) - col_min
                    ymin = max(ymin, row_min) - row_min
                    xmax = min(xmax, col_max) - col_min
                    ymax = min(ymax, row_max) - row_min
                    new_prompt.append(((xmin, ymin, xmax, ymax), lb))
            else:
                raise ValueError(
                    "Invalid prompt format. Expected either a point or a bounding box."
                    f"Got the following prompt instead: {prompt}"
                )

        if is_valid_prompt(new_prompt, n_foreground_points):
            normalized_prompts[prompt_id] = new_prompt

    return normalized_prompts


#
# AUTOMATIC SEGMENTATION METHODS
#


def build_point_grid(points_per_side: int, img_size: int) -> List[Prompt]:
    """Build a grid of points within the image.

    The grid is composed of points spaced evenly across the image, with a total number of points
    equal to points_per_side**2.

    Args:
        points_per_side: Number of points per side.
        img_size: Image size.

    Returns:
        List of points forming the grid.
    """
    offset = img_size / (2 * points_per_side)
    points_one_side = np.linspace(offset, img_size - offset, points_per_side)
    grid_points = [cast(Prompt, ((x, y), 1)) for x, y in product(points_one_side, points_one_side)]
    return grid_points


def generate_crop_boxes(
    chip_size: int, n_layers: int, overlap_ratio: float = 0.0
) -> Tuple[List[BBox], List[int]]:
    """Generate a list of crop boxes of different sizes.

    Each layer has (2**i)**2 boxes for the ith layer.

    Args:
        chip_size: Size of the chip.
        n_layers: Number of layers.
        overlap_ratio: Overlap ratio between crops.
    Returns:
        Tuple of crop boxes and associated layer indices.
    """
    crop_boxes, layer_idxs = [], []

    # Original chip
    crop_boxes.append([0, 0, chip_size, chip_size])
    layer_idxs.append(0)

    def crop_len(orig_len: int, n_crops: int, overlap: int) -> int:
        return int(ceil((overlap * (n_crops - 1) + orig_len) / n_crops))

    for i_layer in range(n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * chip_size * (2 / n_crops_per_side))

        crop_w = crop_len(chip_size, n_crops_per_side, overlap)
        crop_h = crop_len(chip_size, n_crops_per_side, overlap)

        crop_box_x0 = [int((crop_w - overlap) * i) for i in range(n_crops_per_side)]
        crop_box_y0 = [int((crop_h - overlap) * i) for i in range(n_crops_per_side)]

        # Crops in XYWH format
        for x0, y0 in product(crop_box_x0, crop_box_y0):
            box = (x0, y0, min(x0 + crop_w, chip_size), min(y0 + crop_h, chip_size))
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)

    return crop_boxes, layer_idxs


def mask_to_bbox(mask: NDArray[Any]) -> NDArray[Any]:
    """Build the bounding box of a binary mask.

    Args:
        mask: Binary mask.
    Returns:
        Bounding box coordinates (col_min, row_min, col_max, row_max) of the mask.
    """
    bbox = []
    for m in np.squeeze(mask, axis=1):
        rows = np.any(m, axis=1)
        cols = np.any(m, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bbox.append([cmin, rmin, cmax, rmax])
    return np.array(bbox, dtype=np.float32)


def translate_bbox(mask_bbox: NDArray[Any], x_offset: float, y_offset: float) -> NDArray[Any]:
    """Translate a mask bounding box by an offset.

    Args:
        mask_bbox: Mask bounding box.
        x_offset: X offset.
        y_offset: Y offset.
    Returns:
        Translated bounding box.
    """
    offset = [[x_offset, y_offset, x_offset, y_offset]]
    return mask_bbox + offset


def uncrop_masks(
    mask: NDArray[Any], mask_bbox: NDArray[Any], crop_box: BBox, chip_size: int
) -> Tuple[NDArray[Any], NDArray[Any]]:
    """Translate and scale a mask from a crop to the original chip size.

    Args:
        mask: Binary mask.
        mask_bbox: Bounding box of the mask.
        crop_box: Crop box.
        chip_size: Chip size.
    Returns:
        Tuple of translated mask and bounding box numpy arrays.
    """
    x0, y0, x1, y1 = map(int, crop_box)
    crop_width = x1 - x0
    crop_height = y1 - y0
    resized_mask = cast(
        torch.Tensor,
        resize(torch.from_numpy(mask), size=[crop_height, crop_width]),
    )
    pad_x, pad_y = chip_size - crop_width, chip_size - crop_height
    pad = (x0, pad_x - x0, y0, pad_y - y0)

    mask = torch.nn.functional.pad(resized_mask, pad, value=0).numpy()

    scale_x, scale_y = crop_width / chip_size, crop_height / chip_size
    mask_bbox = mask_bbox.astype(np.float64) * np.array([scale_y, scale_x, scale_y, scale_x])
    return mask, np.round(mask_bbox).astype(np.float32)


def calculate_stability_score(
    masks: NDArray[Any], mask_threshold: float, threshold_offset: float
) -> NDArray[Any]:
    """Compute the stability score for a batch of masks.

    The stability score is the IoU between the binary masks obtained by thresholding
    the predicted mask logits at high and low values.

    Args:
        masks: Mask logits.
        mask_threshold: Mask threshold.
        threshold_offset: Threshold offset.

    Returns:
        Stability score.
    """
    intersections = np.sum(masks > (mask_threshold + threshold_offset), axis=(2, 3))
    unions = np.sum(masks > (mask_threshold - threshold_offset), axis=(2, 3))
    return intersections / unions


#
# ENCODER/DECODER PREPROCESSING
#


def build_chip_preprocessing_operation(
    raster: Raster,
    band_names: Optional[List[str]],
    band_scaling: Optional[List[float]],
    band_offset: Optional[List[float]],
) -> Callable[[NDArray[Any]], NDArray[Any]]:
    if band_names:
        if len(band_names) == 1:
            LOGGER.info(
                "Got only a single band name. "
                "Will replicate it to build a 3-channeled chip for SAM."
            )
            band_names = band_names * 3
        elif len(band_names) != 3:
            raise ValueError(
                f"Invalid number of bands. Expected one or three band names. Got {band_names}"
            )
    else:
        LOGGER.info("No bands selected. Using ['R', 'G', 'B']")
        band_names = ["R", "G", "B"]

    if not all([b in raster.bands for b in band_names]):
        raise ValueError(
            f"Band not found in input raster. Expected band names {band_names} "
            f"to be among raster bands {list(raster.bands.keys())}"
        )
    band_idx = [raster.bands[b] for b in band_names]

    if band_scaling:
        if len(band_scaling) == 1:
            LOGGER.info("Got a single scaling parameter. Will use it for all bands.")
            band_scaling = band_scaling * 3
        elif len(band_scaling) != len(band_names):
            raise ValueError(f"Expected one or three scaling parameters. Got {band_scaling}")
    else:
        band_scaling = [raster.scale] * 3
    scale = np.array(band_scaling).reshape(1, 3, 1, 1)

    if band_offset:
        if len(band_offset) == 1:
            LOGGER.info("Got a single offset parameter. Will use it for all bands.")
            band_offset = band_offset * 3
        elif len(band_offset) != len(band_names):
            raise ValueError(f"Expected one or three offset parameters. Got {band_offset}")
    else:
        band_offset = [raster.offset] * 3
    offset = np.array(band_offset).reshape(1, 3, 1, 1)

    def preprocessing_operation(chip: NDArray[Any]) -> NDArray[Any]:
        normalized_chip = chip[:, band_idx, :, :] * scale + offset
        if np.min(normalized_chip) >= 0 and np.max(normalized_chip) <= 1:
            normalized_chip = normalized_chip * 255.0
        return normalized_chip.astype(np.float32)

    return preprocessing_operation


def img_encoder_preprocess(
    chip: NDArray[Any], preprocessing_operation: Callable[[NDArray[Any]], NDArray[Any]]
) -> NDArray[Any]:
    """Preprocesses the input chip for the image encoder model.

    Args:
        chip: Input chip.
        preprocessing_operation: Preprocessing function (depending on the chip type).

    Returns:
        Preprocessed chip.
    """
    processed_chip = preprocessing_operation(chip)
    input_tensor = torch.from_numpy(processed_chip.clip(0, 255))

    # Normalizing input tensor by subtracting pixel mean and dividing by pixel std
    pixel_mean = torch.Tensor(SAM_PIXEL_RGB_MEAN).view(-1, 1, 1)
    pixel_std = torch.Tensor(SAM_PIXEL_RGB_STD).view(-1, 1, 1)
    x = (input_tensor - pixel_mean) / pixel_std
    return x.numpy()


def prompt_encoder_preprocess(
    prompt: List[Prompt],
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Preprocesses the input prompt to the expected decoder format.

    Args:
        prompt: List of prompts.

    Returns:
        Tuple of preprocessed coordinates and labels.
    """
    point_prompt = [p for p in prompt if len(p[0]) == 2]
    bbox_prompt = [p for p in prompt if len(p[0]) == 4]

    if point_prompt:
        coords, labels = zip(*point_prompt)
        point_batch, point_label = np.array(coords), np.array(labels)
    else:
        point_batch, point_label = None, None

    if bbox_prompt:
        coords, _ = zip(*bbox_prompt)
        bbox_batch = np.array(coords).reshape(2, 2)
        bbox_label = np.array([2, 3])
    else:  # Padding with dummy bbox
        bbox_batch = np.array([[0.0, 0.0]])
        bbox_label = np.array([-1])

    onnx_coord = (
        np.concatenate([point_batch, bbox_batch], axis=0)[None, :, :].astype(np.float32)
        if point_batch is not None
        else bbox_batch[None, :, :].astype(np.float32)
    )
    onnx_label = (
        np.concatenate([point_label, bbox_label], axis=0)[None, :].astype(np.float32)
        if point_label is not None
        else bbox_label[None, :].astype(np.float32)
    )

    return onnx_coord, onnx_label


def batch_prompt_encoder_preprocess(
    prompt_group: List[List[Prompt]],
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Preprocesses a batch of prompts for the encoder model.

    Args:
        prompt_group: List of prompt groups.
    Returns:
        Tuple of preprocessed coordinates and labels.
    """
    processed_prompts = [prompt_encoder_preprocess(p) for p in prompt_group]

    onnx_coord = np.concatenate([p[0] for p in processed_prompts], axis=0)
    onnx_label = np.concatenate([p[1] for p in processed_prompts], axis=0)

    return onnx_coord, onnx_label


def mask_encoder_preprocess(
    input_mask: Optional[NDArray[Any]] = None,
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Preprocess the input mask for the encoder model.

    Args:
        input_mask: Input mask.
    Returns:
        Tuple of preprocessed mask and has_mask inputs.
    """
    if not input_mask:
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)
        return onnx_mask_input, onnx_has_mask_input

    # TODO: Implement mask preprocessing if passed as argument
    # input_mask = ...
    return input_mask, np.ones(1, dtype=np.float32)


#
# POSTPROCESSING
#


def get_mask_within_bbox(mask: NDArray[Any], prompt: List[Prompt]) -> NDArray[Any]:
    """Filter input mask pixels only for those within the bounding box of the prompt (if any).

    Args:
        mask: Input mask.
        prompt: List of prompts.
    Returns:
        Mask filtered within the bounding box of the prompt.
    """
    bbox = [coords for coords, _ in prompt if len(coords) == 4]
    if bbox:
        xmin, ymin, xmax, ymax = cast(BBox, bbox[0])
        bbox_mask = np.full(mask.shape, False)
        bbox_mask[
            0, 0, int(round(ymin)) : int(round(ymax)), int(round(xmin)) : int(round(xmax))
        ] = True
        return np.logical_and(mask, bbox_mask)
    return mask


#
# ONNX RUNTIME METHODS
#


def extract_img_embeddings_from_chip(
    chip_data: NDArray[Any],
    preprocessing_operation: Callable[[NDArray[Any]], NDArray[Any]],
    encoder: ort.InferenceSession,
) -> NDArray[Any]:
    """Extract image embeddings from a chip using the encoder model.

    Args:
        chip_data: Input chip data.
        preprocessing_operation: Preprocessing operation for the chip.
        encoder: ONNX encoder model.
    Returns:
        Image embeddings.
    """
    model_input = img_encoder_preprocess(chip_data, preprocessing_operation)
    model_output = encoder.run(None, {encoder.get_inputs()[0].name: model_input})[0]
    return model_output
