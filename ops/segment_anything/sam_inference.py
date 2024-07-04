import logging
import os
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import onnxruntime as ort
import torch
from numpy.typing import NDArray
from shapely import geometry as shpg
from shapely.geometry.base import BaseGeometry
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize

from vibe_core.data import (
    AssetVibe,
    BBox,
    CategoricalRaster,
    ChipWindow,
    GeometryCollection,
    Raster,
    SamMaskRaster,
    gen_guid,
)
from vibe_lib.raster import INT_COMPRESSION_KWARGS, write_window_to_file
from vibe_lib.segment_anything import (
    BACKGROUND_VALUE,
    MASK_LOGIT_THRESHOLD,
    SAM_CHIP_SIZE,
    Prompt,
    batch_prompt_encoder_preprocess,
    build_chip_preprocessing_operation,
    build_point_grid,
    calculate_stability_score,
    extract_img_embeddings_from_chip,
    generate_crop_boxes,
    get_mask_within_bbox,
    get_normalized_prompts_within_chip,
    mask_encoder_preprocess,
    mask_to_bbox,
    preprocess_geometry_collection,
    prompt_encoder_preprocess,
    translate_bbox,
    uncrop_masks,
)
from vibe_lib.spaceeye.chip import (
    ChipDataset,
    ChipDataType,
    Dims,
    InMemoryReader,
    Window,
    get_loader,
    write_prediction_to_file,
)

BASE_MODEL_PATH = "/mnt/onnx_resources/{model_type}_{model_part}.onnx"
SAM_MODEL_TYPES = ["vit_h", "vit_l", "vit_b"]


LOGGER = logging.getLogger(__name__)


class CallbackBuilder:
    def __init__(
        self,
        model_type: str,
        spatial_overlap: float,
        num_workers: int,
        in_memory: bool,
        band_names: Optional[List[str]],
        band_scaling: Optional[List[float]],
        band_offset: Optional[List[float]],
    ):
        self.model_type = model_type
        self.spatial_overlap = spatial_overlap
        self.num_workers = num_workers
        self.in_memory = in_memory
        self.tmp_dir = TemporaryDirectory()
        self.window_size = SAM_CHIP_SIZE
        self.band_names = band_names
        self.band_scaling = band_scaling
        self.band_offset = band_offset

    def get_model(self) -> Tuple[ort.InferenceSession, ort.InferenceSession]:
        if self.model_type not in SAM_MODEL_TYPES:
            raise ValueError(
                f"Unknown model type: '{self.model_type}'. Expected one of {SAM_MODEL_TYPES}"
            )

        encoder_path = BASE_MODEL_PATH.format(model_type=self.model_type, model_part="encoder")
        decoder_path = BASE_MODEL_PATH.format(model_type=self.model_type, model_part="decoder")

        if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
            raise ValueError(
                f"Model files not found for model type: '{self.model_type}'. "
                f"Refer to the troubleshooting section of FarmVibes.AI documentation "
                f"for instructions on how to import the model files to the cluster."
            )

        encoder = ort.InferenceSession(encoder_path)
        LOGGER.info(f"Loaded encoder model from {encoder_path}")
        decoder = ort.InferenceSession(decoder_path)
        LOGGER.info(f"Loaded decoder model from {decoder_path}")
        return encoder, decoder

    def get_chip_dataloader(
        self,
        raster: Raster,
        geometry: BaseGeometry,
    ) -> DataLoader[ChipDataType]:
        chip_size = self.window_size
        step_size = int(chip_size * (1 - self.spatial_overlap))
        dataset = ChipDataset(
            [raster],
            chip_size=Dims(chip_size, chip_size, 1),
            step_size=Dims(step_size, step_size, 1),
            nodata=BACKGROUND_VALUE,
            geometry_or_chunk=geometry,
            reader=InMemoryReader(downsampling=1) if self.in_memory else None,
        )

        dataloader = get_loader(
            dataset, batch_size=1, num_workers=self.num_workers if not self.in_memory else 0
        )

        return dataloader

    def __del__(self):
        self.tmp_dir.cleanup()


class PromptCallbackBuilder(CallbackBuilder):
    img_preprocessing_operation: Callable[[NDArray[Any]], NDArray[Any]]

    def __init__(
        self,
        model_type: str,
        spatial_overlap: float,
        points_per_batch: int,
        num_workers: int,
        in_memory: bool,
        band_names: Optional[List[str]],
        band_scaling: Optional[List[float]],
        band_offset: Optional[List[float]],
    ):
        super().__init__(
            model_type,
            spatial_overlap,
            num_workers,
            in_memory,
            band_names,
            band_scaling,
            band_offset,
        )
        self.points_per_batch = points_per_batch

    def get_mask_for_prompt_group(
        self,
        prompt_group: List[Prompt],
        chip_data: NDArray[Any],
        decoder_session: ort.InferenceSession,
        img_embedding: NDArray[Any],
    ) -> NDArray[Any]:
        prompt_group_mask = np.zeros((1, 1, *chip_data.shape[-2:]), dtype=bool)
        for i in range(0, len(prompt_group), self.points_per_batch):
            prompt_batch, prompt_label = prompt_encoder_preprocess(
                prompt_group[i : i + self.points_per_batch]
            )
            mask_prompt, has_mask_prompt = mask_encoder_preprocess()

            ort_inputs = {
                "image_embeddings": img_embedding,
                "point_coords": prompt_batch,
                "point_labels": prompt_label,
                "mask_input": mask_prompt,
                "has_mask_input": has_mask_prompt,
                "orig_im_size": np.array([self.window_size, self.window_size], dtype=np.float32),
            }

            predicted_mask, _, _ = decoder_session.run(None, ort_inputs)
            predicted_mask = predicted_mask > MASK_LOGIT_THRESHOLD
            prompt_group_mask = np.logical_or(prompt_group_mask, predicted_mask)

        # Only include in the mask, pixels within the prompted bounding box
        prompt_group_mask = get_mask_within_bbox(prompt_group_mask, prompt_group)

        return prompt_group_mask

    def generate_masks_from_points(
        self,
        dataloader: DataLoader[ChipDataType],
        encoder_session: ort.InferenceSession,
        decoder_session: ort.InferenceSession,
        input_prompts: Dict[int, List[Prompt]],
    ) -> List[str]:
        filepaths: List[str] = []
        dataset = cast(ChipDataset, dataloader.dataset)
        get_filename = dataset.get_filename
        for batch_idx, batch in enumerate(dataloader):
            chip_data, chip_mask, write_info_list = batch
            output_chip_mask = np.zeros((1, len(input_prompts), *chip_data.shape[-2:]), dtype=bool)

            prompts_in_chip = get_normalized_prompts_within_chip(
                input_prompts, dataset.read_windows[batch_idx][0], dataset.offset
            )

            if prompts_in_chip:
                LOGGER.info(f"Running model for batch ({batch_idx + 1}/{len(dataloader)})")

                img_embedding = extract_img_embeddings_from_chip(
                    chip_data, self.img_preprocessing_operation, encoder_session
                )

                for prompt_id, prompt_group in prompts_in_chip.items():
                    prompt_group_mask = self.get_mask_for_prompt_group(
                        prompt_group, chip_data, decoder_session, img_embedding
                    )
                    output_chip_mask[0, prompt_id] = np.logical_or(
                        output_chip_mask[0, prompt_id], prompt_group_mask[0, 0]
                    )

            else:
                LOGGER.info(
                    "Skipping batch with no prompt intersection "
                    f"({batch_idx + 1}/{len(dataloader)})"
                )

            write_prediction_to_file(
                output_chip_mask.astype(np.uint8),
                chip_mask,
                write_info_list,
                self.tmp_dir.name,
                filepaths,
                get_filename,
            )

        return filepaths

    def __call__(self):
        def callback(
            input_raster: Raster,
            input_prompts: GeometryCollection,
        ) -> Dict[str, CategoricalRaster]:
            geometry = shpg.shape(input_raster.geometry)
            dataloader = self.get_chip_dataloader(input_raster, geometry)

            processed_prompts, prompt_id_map = preprocess_geometry_collection(
                input_prompts, cast(ChipDataset, dataloader.dataset), geometry
            )

            self.img_preprocessing_operation = build_chip_preprocessing_operation(
                input_raster, self.band_names, self.band_scaling, self.band_offset
            )

            encoder_session, decoder_session = self.get_model()

            mask_filepaths = self.generate_masks_from_points(
                dataloader,
                encoder_session,
                decoder_session,
                processed_prompts,
            )

            asset = AssetVibe(reference=mask_filepaths[0], type="image/tiff", id=gen_guid())
            segmentation_mask = CategoricalRaster.clone_from(
                input_raster,
                id=gen_guid(),
                assets=[asset],
                bands={
                    f"mask_prompt_{prompt_id_map[prompt_id]}": prompt_id
                    for prompt_id in processed_prompts.keys()
                },
                categories=["background", "foreground"],
            )

            return {"segmentation_mask": segmentation_mask}

        return callback


class AutomaticSegmentationCallbackBuilder(PromptCallbackBuilder):
    def __init__(
        self,
        model_type: str,
        spatial_overlap: float,
        points_per_side: int,
        n_crop_layers: int,
        crop_overlap_ratio: float,
        crop_n_points_downscale_factor: int,
        pred_iou_thresh: float,
        stability_score_thresh: float,
        stability_score_offset: float,
        points_per_batch: int,
        num_workers: int,
        in_memory: bool,
        band_names: Optional[List[str]],
        band_scaling: Optional[List[float]],
        band_offset: Optional[List[float]],
    ):
        super().__init__(
            model_type,
            spatial_overlap,
            points_per_batch,
            num_workers,
            in_memory,
            band_names,
            band_scaling,
            band_offset,
        )
        self.points_per_side = points_per_side
        self.n_crop_layers = n_crop_layers
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.validate_parameters()

    def validate_parameters(self):
        if not isinstance(self.points_per_side, int) or self.points_per_side < 1:
            raise ValueError(
                f"'points_per_side' must be a positive integer. Got {self.points_per_side}."
            )
        if not isinstance(self.n_crop_layers, int) or self.n_crop_layers < 0:
            raise ValueError(
                f"'n_crop_layers' must be a non-negative integer. Got {self.n_crop_layers}."
            )
        if self.crop_overlap_ratio < 0 or self.crop_overlap_ratio >= 1:
            raise ValueError(
                "'crop_overlap_ratio' must be a float in the range [0, 1). "
                f"Got {self.crop_overlap_ratio}."
            )
        if (
            not isinstance(self.crop_n_points_downscale_factor, int)
            or self.crop_n_points_downscale_factor < 1
        ):
            raise ValueError(
                "'crop_n_points_downscale_factor' must be a positive integer. "
                f"Got {self.crop_n_points_downscale_factor}."
            )
        if self.pred_iou_thresh <= 0 or self.pred_iou_thresh >= 1:
            raise ValueError(
                "'pred_iou_thresh' must be a float in the range (0, 1). "
                f"Got {self.pred_iou_thresh}."
            )
        if self.stability_score_thresh <= 0 or self.stability_score_thresh > 1:
            raise ValueError(
                "'stability_score_thresh' must be a float in the range (0, 1]. "
                f"Got {self.stability_score_thresh}."
            )

    def point_grid_inference(
        self,
        prompts: List[Prompt],
        img_embedding: NDArray[Any],
        decoder_session: ort.InferenceSession,
    ) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        mask, mask_scores, mask_bbox = [], [], []
        mask_prompt, has_mask_prompt = mask_encoder_preprocess()
        for i in range(0, len(prompts), self.points_per_batch):
            LOGGER.info(
                f"Processing points {i}-{min(i + self.points_per_batch, len(prompts))} "
                f"out of {len(prompts)}"
            )
            batch = [[p] for p in prompts[i : i + self.points_per_batch]]
            prompt_batch, prompt_label = batch_prompt_encoder_preprocess(batch)
            ort_inputs = {
                "image_embeddings": img_embedding,
                "point_coords": prompt_batch,
                "point_labels": prompt_label,
                "mask_input": mask_prompt,
                "has_mask_input": has_mask_prompt,
                "orig_im_size": np.array([self.window_size, self.window_size], dtype=np.float32),
            }
            pred_mask, pred_scores, _ = decoder_session.run(None, ort_inputs)

            # Filter by the mask quality score provided by SAM
            if self.pred_iou_thresh > 0:
                keep_masks = (pred_scores > self.pred_iou_thresh).reshape(-1)
                pred_mask = pred_mask[keep_masks]
                pred_scores = pred_scores[keep_masks]

            # Filter by Stability Score
            if self.stability_score_thresh > 0:
                stability_score = calculate_stability_score(
                    pred_mask, MASK_LOGIT_THRESHOLD, self.stability_score_offset
                )
                keep_masks = (stability_score > self.stability_score_thresh).reshape(-1)
                pred_mask = pred_mask[keep_masks]
                pred_scores = pred_scores[keep_masks]

            if pred_mask.shape[0] > 0:
                # Binarize mask given logit threshold
                pred_mask = pred_mask > MASK_LOGIT_THRESHOLD
                mask.append(pred_mask)
                mask_scores.append(pred_scores.reshape(-1))
                mask_bbox.append(mask_to_bbox(pred_mask))

        mask = np.concatenate(mask, axis=0)
        mask_scores = np.concatenate(mask_scores, axis=0)
        mask_bbox = np.concatenate(mask_bbox, axis=0)
        return mask, mask_scores, mask_bbox

    def process_crop(
        self,
        chip_data: NDArray[Any],
        crop_box: BBox,
        layer_idx: int,
        encoder_session: ort.InferenceSession,
        decoder_session: ort.InferenceSession,
    ) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        # Get crop and resize
        x0, y0, x1, y1 = crop_box
        cropped_im = chip_data[:, :, y0:y1, x0:x1]

        if layer_idx > 0:  # Resize to chip size if not the first layer
            cropped_im = cast(
                torch.Tensor,
                resize(torch.from_numpy(cropped_im), size=[self.window_size]),
            ).numpy()

        # Get crop embeddings
        crop_img_embedding = extract_img_embeddings_from_chip(
            cropped_im, self.img_preprocessing_operation, encoder_session
        )

        # Build point grid for crop
        points_per_side_for_layer = int(
            self.points_per_side / (self.crop_n_points_downscale_factor**layer_idx)
        )
        prompts = build_point_grid(points_per_side_for_layer, self.window_size)

        # Build mask
        mask, mask_scores, mask_bbox = self.point_grid_inference(
            prompts, crop_img_embedding, decoder_session
        )

        if layer_idx > 0:  # Resize mask to crop size if not the first layer
            mask, mask_bbox = uncrop_masks(mask, mask_bbox, crop_box, self.window_size)

        # Return to the original image frame
        mask_bbox = translate_bbox(mask_bbox, x_offset=crop_box[0], y_offset=crop_box[1])

        return mask, mask_scores, mask_bbox

    def generate_masks_from_grid(
        self,
        dataloader: DataLoader[ChipDataType],
        encoder_session: ort.InferenceSession,
        decoder_session: ort.InferenceSession,
    ) -> Tuple[List[str], List[NDArray[Any]], List[NDArray[Any]], List[ChipWindow]]:
        filepaths: List[str] = []
        scores: List[NDArray[Any]] = []
        boxes: List[NDArray[Any]] = []
        chip_windows: List[ChipWindow] = []

        file_id = gen_guid()
        dataset = cast(ChipDataset, dataloader.dataset)

        # Generate smaller crops within each chip (if n_crop_layers > 0)
        crop_boxes, layer_idxs = generate_crop_boxes(
            self.window_size, self.n_crop_layers, self.crop_overlap_ratio
        )

        for batch_idx, batch in enumerate(dataloader):
            LOGGER.info(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
            chip_data, chip_mask, write_info_list = batch
            read_window = dataset.read_windows[batch_idx][0]

            crop_masks, crop_scores, crop_bbox = [], [], []

            # Generate masks for each crop within chip
            for crop_idx, (crop_box, layer_idx) in enumerate(zip(crop_boxes, layer_idxs)):
                LOGGER.info(
                    f"Processing crop {crop_idx + 1}/{len(crop_boxes)} from layer idx {layer_idx}"
                )
                mask, mask_scores, mask_bbox = self.process_crop(
                    chip_data, crop_box, layer_idx, encoder_session, decoder_session
                )
                crop_masks.append(mask)
                crop_scores.append(mask_scores)
                crop_bbox.append(mask_bbox)

            crop_masks = np.concatenate(crop_masks, axis=0)
            crop_scores = np.concatenate(crop_scores, axis=0)
            crop_bbox = np.concatenate(crop_bbox, axis=0)

            # Translate crop_box in relation to input raster
            crop_bbox = translate_bbox(
                crop_bbox, x_offset=read_window.col_off, y_offset=read_window.row_off
            )

            # Write chip to file
            if crop_masks.shape[0] > 0:
                LOGGER.info(f"Writing masks to file {batch_idx + 1}/{len(dataloader)}")
                filename = os.path.join(self.tmp_dir.name, f"{file_id}_{batch_idx}.tif")
                meta = cast(Dict[str, Any], write_info_list[0]["meta"])
                meta.update({**INT_COMPRESSION_KWARGS})

                write_window = ChipWindow(
                    int(read_window.col_off - dataset.offset.width),
                    int(read_window.row_off - dataset.offset.height),
                    int(read_window.width),
                    int(read_window.height),
                )

                write_window_to_file(
                    crop_masks.squeeze(axis=1),
                    chip_mask.any(axis=(0, 1)),
                    Window(*write_window),  # type: ignore
                    filename,
                    meta,
                )
                filepaths.append(filename)
                scores.append(crop_scores)
                boxes.append(crop_bbox)
                chip_windows.append(write_window)
            else:
                LOGGER.info(f"No masks to write from batch {batch_idx + 1}/{len(dataloader)}")

        return filepaths, scores, boxes, chip_windows

    def __call__(self):
        def callback(
            input_raster: Raster,
        ) -> Dict[str, List[SamMaskRaster]]:
            geometry = shpg.shape(input_raster.geometry)
            dataloader = self.get_chip_dataloader(input_raster, geometry)

            self.img_preprocessing_operation = build_chip_preprocessing_operation(
                input_raster, self.band_names, self.band_scaling, self.band_offset
            )

            encoder_session, decoder_session = self.get_model()

            chip_filepaths, mask_scores, mask_boxes, chip_windows = self.generate_masks_from_grid(
                dataloader,
                encoder_session,
                decoder_session,
            )

            rasters: List[SamMaskRaster] = []
            for path, scores, boxes, window in zip(
                chip_filepaths, mask_scores, mask_boxes, chip_windows
            ):
                asset = AssetVibe(reference=path, type="image/tiff", id=gen_guid())
                segmented_chip = SamMaskRaster.clone_from(
                    input_raster,
                    id=gen_guid(),
                    assets=[asset],
                    bands={f"mask_{i}": i for i in range(scores.shape[0])},
                    categories=["background", "foreground"],
                    mask_score=scores.tolist(),
                    mask_bbox=boxes.tolist(),
                    chip_window=window,
                )
                rasters.append(segmented_chip)

            return {"segmented_chips": rasters}

        return callback
