# farm_ai/segmentation/auto_segment_basemap

Downloads basemap with BingMaps API and runs Segment Anything Model (SAM) automatic segmentation over them. The workflow lists and downloads basemaps tiles with BingMaps API, and merges them into a single raster. The raster is then split into chips of 1024x1024 pixels with an overlap defined by `spatial_overlap`. Each chip is processed by SAM's image encoder, and a point grid is defined within each chip, with each point being used as a prompt for the segmentation. Each point is used to generate a mask, and the masks are combined using multiple non-maximal suppression steps to generate the final segmentation mask. Before running the workflow, make sure the model has been imported into the cluster by running `scripts/export_prompt_segmentation_models.py`. The script will download the desired model weights from SAM repository, export the image encoder and mask decoder to ONNX format, and add them to the cluster. For more information, refer to the [FarmVibes.AI troubleshooting](https://microsoft.github.io/farmvibes-ai/docfiles/markdown/TROUBLESHOOTING.html) page in the documentation.

```{mermaid}
    graph TD
    inp1>user_input]
    out1>basemap]
    out2>segmentation_mask]
    tsk1{{basemap_download}}
    tsk2{{basemap_automatic_segmentation}}
    tsk1{{basemap_download}} -- merged_basemap/input_raster --> tsk2{{basemap_automatic_segmentation}}
    inp1>user_input] -- input_geometry --> tsk1{{basemap_download}}
    inp1>user_input] -- input_geometry --> tsk2{{basemap_automatic_segmentation}}
    tsk1{{basemap_download}} -- merged_basemap --> out1>basemap]
    tsk2{{basemap_automatic_segmentation}} -- segmentation_mask --> out2>segmentation_mask]
```

## Sources

- **user_input**: Time range and geometry of interest.

## Sinks

- **basemap**: Merged basemap used as input to the segmentation.

- **segmentation_mask**: Output segmentation masks.

## Parameters

- **bingmaps_api_key**: Required BingMaps API key.

- **basemap_zoom_level**: Zoom level of interest, ranging from 0 to 20. For instance, a zoom level of 1 corresponds to a resolution of 78271.52 m/pixel, a zoom level of 10 corresponds to 152.9 m/pixel, and a zoom level of 19 corresponds to 0.3 m/pixel. For more information on zoom levels and their corresponding scale and resolution, please refer to the BingMaps API documentation at https://learn.microsoft.com/en-us/bingmaps/articles/understanding-scale-and-resolution

- **model_type**: SAM's image encoder backbone architecture, among 'vit_h', 'vit_l', or 'vit_b'. Before running the workflow, make sure the desired model has been exported to the cluster by running `scripts/export_sam_models.py`. For more information, refer to the FarmVibes.AI troubleshooting page in the documentation.

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

- **basemap_download**: Downloads Bing Maps basemap tiles and merges them into a single raster.

- **basemap_automatic_segmentation**: Runs a Segment Anything Model (SAM) automatic segmentation over input rasters.

## Workflow Yaml

```yaml

name: auto_segment_basemap
sources:
  user_input:
  - basemap_download.input_geometry
  - basemap_automatic_segmentation.input_geometry
sinks:
  basemap: basemap_download.merged_basemap
  segmentation_mask: basemap_automatic_segmentation.segmentation_mask
parameters:
  bingmaps_api_key: null
  basemap_zoom_level: 14
  model_type: vit_b
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
  basemap_download:
    workflow: data_ingestion/bing/basemap_download_merge
    parameters:
      api_key: '@from(bingmaps_api_key)'
      zoom_level: '@from(basemap_zoom_level)'
  basemap_automatic_segmentation:
    workflow: ml/segment_anything/automatic_segmentation
    parameters:
      model_type: '@from(model_type)'
      band_names:
      - red
      - green
      - blue
      band_scaling: null
      band_offset: null
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
      chip_nms_thr: '@from(chip_nms_thr)'
      mask_nms_thr: '@from(mask_nms_thr)'
edges:
- origin: basemap_download.merged_basemap
  destination:
  - basemap_automatic_segmentation.input_raster
description:
  short_description: Downloads basemap with BingMaps API and runs Segment Anything
    Model (SAM) automatic segmentation over them.
  long_description: The workflow lists and downloads basemaps tiles with BingMaps
    API, and merges them into a single raster. The raster is then split into chips
    of 1024x1024 pixels with an overlap defined by `spatial_overlap`. Each chip is
    processed by SAM's image encoder, and a point grid is defined within each chip,
    with each point being used as a prompt for the segmentation. Each point is used
    to generate a mask, and the masks are combined using multiple non-maximal suppression
    steps to generate the final segmentation mask. Before running the workflow, make
    sure the model has been imported into the cluster by running `scripts/export_prompt_segmentation_models.py`.
    The script will download the desired model weights from SAM repository, export
    the image encoder and mask decoder to ONNX format, and add them to the cluster.
    For more information, refer to the [FarmVibes.AI troubleshooting](https://microsoft.github.io/farmvibes-ai/docfiles/markdown/TROUBLESHOOTING.html)
    page in the documentation.
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    basemap: Merged basemap used as input to the segmentation.
    segmentation_mask: Output segmentation masks.


```