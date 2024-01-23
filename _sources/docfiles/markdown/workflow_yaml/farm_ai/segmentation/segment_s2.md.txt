# farm_ai/segmentation/segment_s2

```yaml

name: segment_s2
sources:
  user_input:
  - preprocess_s2.user_input
  - sam_inference.input_geometry
  prompts:
  - ingest_points.user_input
sinks:
  s2_raster: preprocess_s2.raster
  segmentation_mask: sam_inference.segmentation_mask
parameters:
  model_type: vit_b
  spatial_overlap: 0.5
  pc_key: null
tasks:
  preprocess_s2:
    workflow: data_ingestion/sentinel2/preprocess_s2
    parameters:
      pc_key: '@from(pc_key)'
  ingest_points:
    workflow: data_ingestion/user_data/ingest_geometry
  sam_inference:
    op: s2_prompt_segmentation
    op_dir: segment_anything
    parameters:
      model_type: '@from(model_type)'
      spatial_overlap: '@from(spatial_overlap)'
edges:
- origin: preprocess_s2.raster
  destination:
  - sam_inference.input_raster
- origin: ingest_points.geometry
  destination:
  - sam_inference.input_prompts
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

```{mermaid}
    graph TD
    inp1>user_input]
    inp2>prompts]
    out1>s2_raster]
    out2>segmentation_mask]
    tsk1{{preprocess_s2}}
    tsk2{{ingest_points}}
    tsk3{{sam_inference}}
    tsk1{{preprocess_s2}} -- raster/input_raster --> tsk3{{sam_inference}}
    tsk2{{ingest_points}} -- geometry/input_prompts --> tsk3{{sam_inference}}
    inp1>user_input] -- user_input --> tsk1{{preprocess_s2}}
    inp1>user_input] -- input_geometry --> tsk3{{sam_inference}}
    inp2>prompts] -- user_input --> tsk2{{ingest_points}}
    tsk1{{preprocess_s2}} -- raster --> out1>s2_raster]
    tsk3{{sam_inference}} -- segmentation_mask --> out2>segmentation_mask]
```