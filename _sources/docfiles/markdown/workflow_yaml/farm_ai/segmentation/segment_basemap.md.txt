# farm_ai/segmentation/segment_basemap

```yaml

name: segment_basemap
sources:
  user_input:
  - basemap_download.input_geometry
  - sam_inference.input_geometry
  prompts:
  - ingest_points.user_input
sinks:
  basemap: basemap_download.merged_basemap
  segmentation_mask: sam_inference.segmentation_mask
parameters:
  bingmaps_api_key: null
  basemap_zoom_level: 14
  model_type: vit_b
  spatial_overlap: 0.5
tasks:
  basemap_download:
    workflow: data_ingestion/bing/basemap_download_merge
    parameters:
      api_key: '@from(bingmaps_api_key)'
      zoom_level: '@from(basemap_zoom_level)'
  ingest_points:
    workflow: data_ingestion/user_data/ingest_geometry
  sam_inference:
    op: basemap_prompt_segmentation
    op_dir: segment_anything
    parameters:
      model_type: '@from(model_type)'
      spatial_overlap: '@from(spatial_overlap)'
edges:
- origin: basemap_download.merged_basemap
  destination:
  - sam_inference.input_raster
- origin: ingest_points.geometry
  destination:
  - sam_inference.input_prompts
description:
  short_description: Downloads basemap with BingMaps API and runs Segment Anything
    Model (SAM) over them with points and/or bounding boxes as prompts.
  long_description: The workflow lists and downloads basemaps tiles with BingMaps
    API, and merges them into a single raster. The raster is then split into chips
    of 1024x1024 pixels with an overlap defined by `spatial_overlap`. Chips intersecting
    with prompts are processed by SAM's image encoder, followed by prompt encoder
    and mask decoder. Before running the workflow, make sure the model has been imported
    into the cluster by running `scripts/export_prompt_segmentation_models.py`. The
    script will download the desired model weights from SAM repository, export the
    image encoder and mask decoder to ONNX format, and add them to the cluster. For
    more information, refer to the [FarmVibes.AI troubleshooting](https://microsoft.github.io/farmvibes-ai/docfiles/markdown/TROUBLESHOOTING.html)
    page in the documentation.
  sources:
    user_input: Time range and geometry of interest.
    prompts: ExternalReferences to the point and/or bounding box prompts. These are
      GeoJSON with coordinates, label (foreground/background) and prompt id (in case,
      the raster contains multiple entities that should be segmented in a single workflow
      run).
  sinks:
    basemap: Merged basemap used as input to the segmentation.
    segmentation_mask: Output segmentation masks.


```

```{mermaid}
    graph TD
    inp1>user_input]
    inp2>prompts]
    out1>basemap]
    out2>segmentation_mask]
    tsk1{{basemap_download}}
    tsk2{{ingest_points}}
    tsk3{{sam_inference}}
    tsk1{{basemap_download}} -- merged_basemap/input_raster --> tsk3{{sam_inference}}
    tsk2{{ingest_points}} -- geometry/input_prompts --> tsk3{{sam_inference}}
    inp1>user_input] -- input_geometry --> tsk1{{basemap_download}}
    inp1>user_input] -- input_geometry --> tsk3{{sam_inference}}
    inp2>prompts] -- user_input --> tsk2{{ingest_points}}
    tsk1{{basemap_download}} -- merged_basemap --> out1>basemap]
    tsk3{{sam_inference}} -- segmentation_mask --> out2>segmentation_mask]
```