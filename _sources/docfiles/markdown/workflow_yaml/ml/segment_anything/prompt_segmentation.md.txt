# ml/segment_anything/prompt_segmentation

Runs Segment Anything Model (SAM) over input rasters with points and/or bounding boxes as prompts. The workflow splits the input input rasters into chips of 1024x1024 pixels with an overlap defined by `spatial_overlap`. Chips intersecting with prompts are processed by SAM's image encoder, followed by prompt encoder and mask decoder. Before running the workflow, make sure the model has been imported into the cluster by running `scripts/export_prompt_segmentation_models.py`. The script will download the desired model weights from SAM repository, export the image encoder and mask decoder to ONNX format, and add them to the cluster. For more information, refer to the [FarmVibes.AI troubleshooting](https://microsoft.github.io/farmvibes-ai/docfiles/markdown/TROUBLESHOOTING.html) page in the documentation.

```{mermaid}
    graph TD
    inp1>input_raster]
    inp2>input_geometry]
    inp3>input_prompts]
    out1>segmentation_mask]
    tsk1{{ingest_points}}
    tsk2{{clip}}
    tsk3{{sam_inference}}
    tsk1{{ingest_points}} -- geometry/input_prompts --> tsk3{{sam_inference}}
    tsk2{{clip}} -- clipped_raster/input_raster --> tsk3{{sam_inference}}
    inp1>input_raster] -- raster --> tsk2{{clip}}
    inp2>input_geometry] -- input_geometry --> tsk2{{clip}}
    inp3>input_prompts] -- user_input --> tsk1{{ingest_points}}
    tsk3{{sam_inference}} -- segmentation_mask --> out1>segmentation_mask]
```

## Sources

- **input_geometry**: Geometry of interest within the raster for the segmentation.

- **input_raster**: Rasters used as input for the segmentation.

- **input_prompts**: ExternalReferences to the point and/or bounding box prompts. These are GeoJSON with coordinates, label (foreground/background) and prompt id (in case, the raster contains multiple entities that should be segmented in a single workflow run).

## Sinks

- **segmentation_mask**: Output segmentation masks.

## Parameters

- **model_type**: SAM's image encoder backbone architecture, among 'vit_h', 'vit_l', or 'vit_b'. Before running the workflow, make sure the desired model has been exported to the cluster by running `scripts/export_sam_models.py`. For more information, refer to the FarmVibes.AI troubleshooting page in the documentation.

- **band_names**: Name of raster bands that should be selected to compose the 3-channel images expected by SAM. If not provided, will try to use ["R", "G", "B"]. If only a single band name is provided, will replicate it through all three channels.

- **band_scaling**: A list of floats to scale each band by to the range of [0.0, 1.0] or [0.0, 255.0]. If not provided, will default to the raster scaling parameter. If a list with a single value is provided, will use it for all three bands.

- **band_offset**: A list of floats to offset each band by. If not provided, will default to the raster offset value. If a list with a single value is provided, will use it for all three bands.

- **spatial_overlap**: Percentage of spatial overlap between chips in the range of [0.0, 1.0).

## Tasks

- **ingest_points**: Adds user geometries into the cluster storage, allowing for them to be used on workflows.

- **clip**: Performs a clip on an input raster based on a provided reference geometry.

- **sam_inference**: Runs SAM over the input raster with points and bounding boxes as prompts.

## Workflow Yaml

```yaml

name: prompt_segmentation
sources:
  input_raster:
  - clip.raster
  input_geometry:
  - clip.input_geometry
  input_prompts:
  - ingest_points.user_input
sinks:
  segmentation_mask: sam_inference.segmentation_mask
parameters:
  model_type: vit_b
  band_names: null
  band_scaling: null
  band_offset: null
  spatial_overlap: 0.5
tasks:
  ingest_points:
    workflow: data_ingestion/user_data/ingest_geometry
  clip:
    workflow: data_processing/clip/clip
  sam_inference:
    op: prompt_segmentation
    op_dir: segment_anything
    parameters:
      model_type: '@from(model_type)'
      band_names: '@from(band_names)'
      band_scaling: '@from(band_scaling)'
      band_offset: '@from(band_offset)'
      spatial_overlap: '@from(spatial_overlap)'
edges:
- origin: ingest_points.geometry
  destination:
  - sam_inference.input_prompts
- origin: clip.clipped_raster
  destination:
  - sam_inference.input_raster
description:
  short_description: Runs Segment Anything Model (SAM) over input rasters with points
    and/or bounding boxes as prompts.
  long_description: The workflow splits the input input rasters into chips of 1024x1024
    pixels with an overlap defined by `spatial_overlap`. Chips intersecting with prompts
    are processed by SAM's image encoder, followed by prompt encoder and mask decoder.
    Before running the workflow, make sure the model has been imported into the cluster
    by running `scripts/export_prompt_segmentation_models.py`. The script will download
    the desired model weights from SAM repository, export the image encoder and mask
    decoder to ONNX format, and add them to the cluster. For more information, refer
    to the [FarmVibes.AI troubleshooting](https://microsoft.github.io/farmvibes-ai/docfiles/markdown/TROUBLESHOOTING.html)
    page in the documentation.
  sources:
    input_geometry: Geometry of interest within the raster for the segmentation.
    input_raster: Rasters used as input for the segmentation.
    input_prompts: ExternalReferences to the point and/or bounding box prompts. These
      are GeoJSON with coordinates, label (foreground/background) and prompt id (in
      case, the raster contains multiple entities that should be segmented in a single
      workflow run).
  sinks:
    segmentation_mask: Output segmentation masks.


```