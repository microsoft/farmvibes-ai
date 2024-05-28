# farm_ai/segmentation/segment_s2

Downloads Sentinel-2 imagery and runs Segment Anything Model (SAM) over them with points and/or bounding boxes as prompts. The workflow retrieves the relevant Sentinel-2 products with the Planetary Computer (PC) API, and splits the input rasters into chips of 1024x1024 pixels with an overlap defined by `spatial_overlap`. Chips intersecting with prompts are processed by SAM's image encoder, followed by prompt encoder and mask decoder. Before running the workflow, make sure the model has been imported into the cluster by running `scripts/export_prompt_segmentation_models.py`. The script will download the desired model weights from SAM repository, export the image encoder and mask decoder to ONNX format, and add them to the cluster. For more information, refer to the [FarmVibes.AI troubleshooting](https://microsoft.github.io/farmvibes-ai/docfiles/markdown/TROUBLESHOOTING.html) page in the documentation.

```{mermaid}
    graph TD
    inp1>user_input]
    inp2>prompts]
    out1>s2_raster]
    out2>segmentation_mask]
    tsk1{{preprocess_s2}}
    tsk2{{s2_segmentation}}
    tsk1{{preprocess_s2}} -- raster/input_raster --> tsk2{{s2_segmentation}}
    inp1>user_input] -- user_input --> tsk1{{preprocess_s2}}
    inp1>user_input] -- input_geometry --> tsk2{{s2_segmentation}}
    inp2>prompts] -- input_prompts --> tsk2{{s2_segmentation}}
    tsk1{{preprocess_s2}} -- raster --> out1>s2_raster]
    tsk2{{s2_segmentation}} -- segmentation_mask --> out2>segmentation_mask]
```

## Sources

- **user_input**: Time range and geometry of interest.

- **prompts**: ExternalReferences to the point and/or bounding box prompts. These are GeoJSON with coordinates, label (foreground/background) and prompt id (in case, the raster contains multiple entities that should be segmented in a single workflow run).

## Sinks

- **s2_raster**: Sentinel-2 rasters used as input for the segmentation.

- **segmentation_mask**: Output segmentation masks.

## Parameters

- **model_type**: SAM's image encoder backbone architecture, among 'vit_h', 'vit_l', or 'vit_b'. Before running the workflow, make sure the desired model has been exported to the cluster by running `scripts/export_sam_models.py`. For more information, refer to the FarmVibes.AI troubleshooting page in the documentation.

- **spatial_overlap**: Percentage of spatial overlap between chips in the range of [0.0, 1.0).

- **pc_key**: Optional Planetary Computer API key.

## Tasks

- **preprocess_s2**: Downloads and preprocesses Sentinel-2 imagery that covers the input geometry and time range.

- **s2_segmentation**: Runs Segment Anything Model (SAM) over input rasters with points and/or bounding boxes as prompts.

## Workflow Yaml

```yaml

name: segment_s2
sources:
  user_input:
  - preprocess_s2.user_input
  - s2_segmentation.input_geometry
  prompts:
  - s2_segmentation.input_prompts
sinks:
  s2_raster: preprocess_s2.raster
  segmentation_mask: s2_segmentation.segmentation_mask
parameters:
  model_type: vit_b
  spatial_overlap: 0.5
  pc_key: null
tasks:
  preprocess_s2:
    workflow: data_ingestion/sentinel2/preprocess_s2
    parameters:
      pc_key: '@from(pc_key)'
  s2_segmentation:
    workflow: ml/segment_anything/prompt_segmentation
    parameters:
      model_type: '@from(model_type)'
      band_names:
      - R
      - G
      - B
      band_scaling: null
      band_offset: null
      spatial_overlap: '@from(spatial_overlap)'
edges:
- origin: preprocess_s2.raster
  destination:
  - s2_segmentation.input_raster
description:
  short_description: Downloads Sentinel-2 imagery and runs Segment Anything Model
    (SAM) over them with points and/or bounding boxes as prompts.
  long_description: The workflow retrieves the relevant Sentinel-2 products with the
    Planetary Computer (PC) API, and splits the input rasters into chips of 1024x1024
    pixels with an overlap defined by `spatial_overlap`. Chips intersecting with prompts
    are processed by SAM's image encoder, followed by prompt encoder and mask decoder.
    Before running the workflow, make sure the model has been imported into the cluster
    by running `scripts/export_prompt_segmentation_models.py`. The script will download
    the desired model weights from SAM repository, export the image encoder and mask
    decoder to ONNX format, and add them to the cluster. For more information, refer
    to the [FarmVibes.AI troubleshooting](https://microsoft.github.io/farmvibes-ai/docfiles/markdown/TROUBLESHOOTING.html)
    page in the documentation.
  sources:
    user_input: Time range and geometry of interest.
    prompts: ExternalReferences to the point and/or bounding box prompts. These are
      GeoJSON with coordinates, label (foreground/background) and prompt id (in case,
      the raster contains multiple entities that should be segmented in a single workflow
      run).
  sinks:
    s2_raster: Sentinel-2 rasters used as input for the segmentation.
    segmentation_mask: Output segmentation masks.


```