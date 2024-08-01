import os
from tempfile import TemporaryDirectory
from typing import Dict, List, Tuple

import numpy as np
import rasterio
import torch
from torchvision.ops.boxes import batched_nms, box_area

from vibe_core.data import AssetVibe, BBox, CategoricalRaster, ChipWindow, SamMaskRaster, gen_guid


def touch_chip_boundaries(bbox: BBox, chip_window: ChipWindow) -> bool:
    return (
        bbox[0] <= chip_window[0]  # col_offset
        or bbox[1] <= chip_window[1]  # row_offset
        or bbox[2] >= chip_window[0] + chip_window[2]  # col_offset + width
        or bbox[3] >= chip_window[1] + chip_window[3]  # row_offset + height
    )


def is_contained_by_others(current_bbox: BBox, other_boxes: List[BBox], eps: int = 5) -> bool:
    for bbox in other_boxes:
        if (
            current_bbox[0] >= bbox[0] - eps
            and current_bbox[1] >= bbox[1] - eps
            and current_bbox[2] <= bbox[2] + eps
            and current_bbox[3] <= bbox[3] + eps
        ):
            return True
    return False


# - ☑️ Filter masks that touch crop boundaries, but do not touch chip boundaries
# - ❌ NMS of all masks within a crop. I don't think this makes much sense anymore
# - ☑️ NMS for all crops within a chip
# - ❓ Remove small disconnected regions and holdes in a mask, then NMS again
# - ☑️ NMS masks from different chips
def select_masks(
    boxes: List[List[BBox]],
    scores: List[List[float]],
    chip_windows: List[ChipWindow],
    chip_nms_thr: float,
    mask_nms_thr: float,
) -> List[List[int]]:
    # NMS within each chip (using SAM prediction scores)
    kept_idx = []
    for chip_boxes, chip_scores in zip(boxes, scores):
        keep_by_nms = batched_nms(
            boxes=torch.from_numpy(np.array(chip_boxes)).to(torch.float32),
            scores=torch.from_numpy(np.array(chip_scores)).to(torch.float32),
            idxs=torch.zeros(len(chip_boxes)),
            iou_threshold=chip_nms_thr,
        )
        kept_idx.append(keep_by_nms.numpy().tolist())

    # NMS across chips (prefering smaller masks)
    idx_map = [
        (cidx, idx) for cidx, chip_idxs in enumerate(kept_idx) for idx in range(len(chip_idxs))
    ]

    kept_boxes = np.array(
        [
            boxes[chip_idx][to_keep_idx]
            for chip_idx in range(len(kept_idx))
            for to_keep_idx in kept_idx[chip_idx]
        ]
    )

    # As in SAM, prefer smaller masks
    area_scores = 1 / box_area(torch.from_numpy(kept_boxes))

    keep_by_nms = batched_nms(
        boxes=torch.from_numpy(kept_boxes),
        scores=area_scores,
        idxs=torch.zeros(kept_boxes.shape[0]),
        iou_threshold=mask_nms_thr,
    )

    idx_map = [idx_map[idx] for idx in keep_by_nms.numpy().tolist()]
    filtered_mask_idxs = [[] for _ in range(len(boxes))]
    for cidx, idx in idx_map:
        filtered_mask_idxs[cidx].append(kept_idx[cidx][idx])

    # Removing masks that touch their chip boundary and are contained within other masks
    mask_idx_to_keep = [[] for _ in range(len(boxes))]
    for chip_idx, mask_idxs in enumerate(filtered_mask_idxs):
        if mask_idxs:
            other_boxes = [
                boxes[cidx][idx]
                for cidx in range(len(boxes))
                for idx in filtered_mask_idxs[cidx]
                if cidx != chip_idx
            ]
            for idx in mask_idxs:
                if not (
                    touch_chip_boundaries(boxes[chip_idx][idx], chip_windows[chip_idx])
                    and is_contained_by_others(boxes[chip_idx][idx], other_boxes)
                ):
                    mask_idx_to_keep[chip_idx].append(idx)
    return mask_idx_to_keep


def merge_masks(
    masks: List[SamMaskRaster], mask_idx_to_keep: List[List[int]], tmp_dir: str
) -> Tuple[AssetVibe, int]:
    n_masks = sum([len(idxs) for idxs in mask_idx_to_keep])
    with rasterio.open(masks[0].assets[0].path_or_url) as src:
        out_meta = src.meta
        out_meta["count"] = n_masks

    out_path = os.path.join(tmp_dir, f"{gen_guid()}.tif")
    band_idx_to_write = 1
    with rasterio.open(out_path, "w", **out_meta) as dst:
        for raster, idxs in zip(masks, mask_idx_to_keep):
            if idxs:
                with rasterio.open(raster.assets[0].path_or_url) as src:
                    for i in idxs:
                        dst.write(src.read(i + 1), band_idx_to_write)
                        band_idx_to_write += 1

    asset = AssetVibe(reference=out_path, type="image/tiff", id=gen_guid())
    return asset, n_masks


class CallbackBuilder:
    def __init__(self, chip_nms_thr: float, mask_nms_thr: float):
        self.tmp_dir = TemporaryDirectory()

        if chip_nms_thr <= 0 or chip_nms_thr >= 1:
            raise ValueError(f"'chip_nms_thr' must be between 0 and 1. Got {chip_nms_thr}")
        if mask_nms_thr <= 0 or mask_nms_thr >= 1:
            raise ValueError(f"'mask_nms_thr' must be between 0 and 1. Got {mask_nms_thr}")

        self.chip_nms_thr = chip_nms_thr
        self.mask_nms_thr = mask_nms_thr

    def __call__(self):
        def callback(input_masks: List[SamMaskRaster]) -> Dict[str, CategoricalRaster]:
            mask_scores = [m.mask_score for m in input_masks]
            mask_bboxes = [m.mask_bbox for m in input_masks]
            chip_windows = [m.chip_window for m in input_masks]

            mask_idx_to_keep = select_masks(
                mask_bboxes, mask_scores, chip_windows, self.chip_nms_thr, self.mask_nms_thr
            )

            asset, n_masks = merge_masks(input_masks, mask_idx_to_keep, self.tmp_dir.name)
            segmentation_mask = CategoricalRaster.clone_from(
                input_masks[0],
                id=gen_guid(),
                assets=[asset],
                bands={f"mask_{i}": i for i in range(n_masks)},
                categories=["background", "foreground"],
            )
            return {"output_mask": segmentation_mask}

        return callback

    def __del__(self):
        self.tmp_dir.cleanup()
