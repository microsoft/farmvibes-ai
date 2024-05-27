# ml/segment_anything/automatic_segmentation

Runs a Segment Anything Model (SAM) automatic segmentation over input rasters. The workflow splits the input rasters into chips of 1024x1024 pixels with an overlap defined by `spatial_overlap`. Each chip is processed by SAM's image encoder, and a point grid is defined within each chip, with each point being used as a prompt for the segmentation.  Each point is used to generate a mask, and the masks are combined using multiple non-maximal suppression steps to generate the final segmentation mask. Before running the workflow, make sure the model has been imported into the cluster by running `scripts/export_prompt_segmentation_models.py`. The script will download the desired model weights from SAM repository, export the image encoder and mask decoder to ONNX format, and add them to the cluster. For more information, refer to the [FarmVibes.AI troubleshooting](https://microsoft.github.io/farmvibes-ai/docfiles/markdown/TROUBLESHOOTING.html) page in the documentation.

```{mermaid}
    graph TD
    inp1>input_raster]
    inp2>input_geometry]
    out1>segmentation_mask]
    tsk1{{clip}}
    tsk2{{sam_inference}}
    tsk3{{combine_masks}}
    tsk1{{clip}} -- clipped_raster/input_raster --> tsk2{{sam_inference}}
    tsk2{{sam_inference}} -- segmented_chips/input_masks --> tsk3{{combine_masks}}
    inp1>input_raster] -- raster --> tsk1{{clip}}
    inp2>input_geometry] -- input_geometry --> tsk1{{clip}}
    tsk3{{combine_masks}} -- output_mask --> out1>segmentation_mask]
```

## Sources

- **input_raster**: Rasters used as input for the segmentation.

- **input_geometry**: Geometry of interest within the raster for the segmentation.

## Sinks

- **segmentation_mask**: Output segmentation masks.

## Parameters

- **model_type**: SAM's image encoder backbone architecture, among 'vit_h', 'vit_l', or 'vit_b'. Before running the workflow, make sure the desired model has been exported to the cluster by running `scripts/export_sam_models.py`. For more information, refer to the FarmVibes.AI troubleshooting page in the documentation.

- **band_names**: Name of raster bands that should be selected to compose the 3-channel images expected by SAM. If not provided, will try to use ["R", "G", "B"]. If only a single band name is provided, will replicate it through all three channels.

- **band_scaling**: A list of floats to scale each band by to the range of [0.0, 1.0] or [0.0, 255.0]. If not provided, will default to the raster scaling parameter. If a list with a single value is provided, will use it for all three bands.

- **band_offset**: A list of floats to offset each band by. If not provided, will default to the raster offset value. If a list with a single value is provided, will use it for all three bands.

- **spatial_overlap**: Percentage of spatial overlap between chips in the range of [0.0, 1.0).

- **points_per_side**: The number of points to be sampled along one side of the chip to be prompts. The total number of points is points_per_side**2.

- **n_crop_layers**: If >0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where each layer has 2**i_layer number of image crops.

- **crop_overlap_ratio**: Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of the chip length. Later layers with more crops scale down this overlap.

- **crop_n_points_downscale_factor**: The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.

- **pred_iou_thresh**: A filtering threshold in [0,1] over the model's predicted mask quality/score.

- **stability_score_thresh**: A filtering threshold in [0,1], using the stability of the mask under changes to the cutoff used to binarize the model's mask predictions.

- **stability_score_offset**: The amount to shift the cutoff when calculated the stability score.

- **points_per_batch**: Number of points to process in a single batch.

- **num_workers**: Number of workers to use for parallel processing.

- **in_memory**: Whether to load the whole raster in memory when running predictions. Uses more memory (~4GB/worker) but speeds up inference for fast models.

- **chip_nms_thr**: The box IoU cutoff used by non-maximal suppression to filter duplicate masks within a chip.

- **mask_nms_thr**: The box IoU cutoff used by non-maximal suppression to filter duplicate masks between different chips.

## Tasks

- **clip**: Performs a clip on an input raster based on a provided reference geometry.

- **sam_inference**: Runs a SAM automatic segmentation inference over the input raster, generating masks for each chip.

- **combine_masks**: Process intermediary segmentation masks, filtering out duplicates and combining into final mask raster.

## Workflow Yaml

```yaml

name: automatic_segmentation
sources:
  input_raster:
  - clip.raster
  input_geometry:
  - clip.input_geometry
sinks:
  segmentation_mask: combine_masks.output_mask
parameters:
  model_type: vit_b
  band_names: null
  band_scaling: null
  band_offset: null
  spatial_overlap: 0.5
  points_per_side: 16
  n_crop_layers: 0
  crop_overlap_ratio: 0.0
  crop_n_points_downscale_factor: 1
  pred_iou_thresh: 0.88
  stability_score_thresh: 0.95
  stability_score_offset: 1.0
  points_per_batch: 16
  num_workers: 0
  in_memory: true
  chip_nms_thr: 0.7
  mask_nms_thr: 0.5
tasks:
  clip:
    workflow: data_processing/clip/clip
  sam_inference:
    op: automatic_segmentation
    op_dir: segment_anything
    parameters:
      model_type: '@from(model_type)'
      band_names: '@from(band_names)'
      band_scaling: '@from(band_scaling)'
      band_offset: '@from(band_offset)'
      spatial_overlap: '@from(spatial_overlap)'
      points_per_side: '@from(points_per_side)'
      n_crop_layers: '@from(n_crop_layers)'
      crop_overlap_ratio: '@from(crop_overlap_ratio)'
      crop_n_points_downscale_factor: '@from(crop_n_points_downscale_factor)'
      pred_iou_thresh: '@from(pred_iou_thresh)'
      stability_score_thresh: '@from(stability_score_thresh)'
      stability_score_offset: '@from(stability_score_offset)'
      points_per_batch: '@from(points_per_batch)'
      num_workers: '@from(num_workers)'
      in_memory: '@from(in_memory)'
  combine_masks:
    op: combine_sam_masks
    op_dir: segment_anything_combine_masks
    parameters:
      chip_nms_thr: '@from(chip_nms_thr)'
      mask_nms_thr: '@from(mask_nms_thr)'
edges:
- origin: clip.clipped_raster
  destination:
  - sam_inference.input_raster
- origin: sam_inference.segmented_chips
  destination:
  - combine_masks.input_masks
description:
  short_description: Runs a Segment Anything Model (SAM) automatic segmentation over
    input rasters.
  long_description: The workflow splits the input rasters into chips of 1024x1024
    pixels with an overlap defined by `spatial_overlap`. Each chip is processed by
    SAM's image encoder, and a point grid is defined within each chip, with each point
    being used as a prompt for the segmentation.  Each point is used to generate a
    mask, and the masks are combined using multiple non-maximal suppression steps
    to generate the final segmentation mask. Before running the workflow, make sure
    the model has been imported into the cluster by running `scripts/export_prompt_segmentation_models.py`.
    The script will download the desired model weights from SAM repository, export
    the image encoder and mask decoder to ONNX format, and add them to the cluster.
    For more information, refer to the [FarmVibes.AI troubleshooting](https://microsoft.github.io/farmvibes-ai/docfiles/markdown/TROUBLESHOOTING.html)
    page in the documentation.
  sources:
    input_raster: Rasters used as input for the segmentation.
    input_geometry: Geometry of interest within the raster for the segmentation.
  sinks:
    segmentation_mask: Output segmentation masks.


```