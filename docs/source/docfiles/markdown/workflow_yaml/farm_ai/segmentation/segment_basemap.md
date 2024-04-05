# farm_ai/segmentation/segment_basemap

Downloads basemap with BingMaps API and runs Segment Anything Model (SAM) over them with points and/or bounding boxes as prompts. The workflow lists and downloads basemaps tiles with BingMaps API, and merges them into a single raster. The raster is then split into chips of 1024x1024 pixels with an overlap defined by `spatial_overlap`. Chips intersecting with prompts are processed by SAM's image encoder, followed by prompt encoder and mask decoder. Before running the workflow, make sure the model has been imported into the cluster by running `scripts/export_prompt_segmentation_models.py`. The script will download the desired model weights from SAM repository, export the image encoder and mask decoder to ONNX format, and add them to the cluster. For more information, refer to the [FarmVibes.AI troubleshooting](https://microsoft.github.io/farmvibes-ai/docfiles/markdown/TROUBLESHOOTING.html) page in the documentation.

```{mermaid}
    graph TD
    inp1>user_input]
    inp2>prompts]
    out1>basemap]
    out2>segmentation_mask]
    tsk1{{basemap_download}}
    tsk2{{basemap_segmentation}}
    tsk1{{basemap_download}} -- merged_basemap/input_raster --> tsk2{{basemap_segmentation}}
    inp1>user_input] -- input_geometry --> tsk1{{basemap_download}}
    inp1>user_input] -- input_geometry --> tsk2{{basemap_segmentation}}
    inp2>prompts] -- input_prompts --> tsk2{{basemap_segmentation}}
    tsk1{{basemap_download}} -- merged_basemap --> out1>basemap]
    tsk2{{basemap_segmentation}} -- segmentation_mask --> out2>segmentation_mask]
```

## Sources

- **user_input**: Time range and geometry of interest.

- **prompts**: ExternalReferences to the point and/or bounding box prompts. These are GeoJSON with coordinates, label (foreground/background) and prompt id (in case, the raster contains multiple entities that should be segmented in a single workflow run).

## Sinks

- **basemap**: Merged basemap used as input to the segmentation.

- **segmentation_mask**: Output segmentation masks.

## Parameters

- **bingmaps_api_key**: Required BingMaps API key.

- **basemap_zoom_level**: Zoom level of interest, ranging from 0 to 20. For instance, a zoom level of 1 corresponds to a resolution of 78271.52 m/pixel, a zoom level of 10 corresponds to 152.9 m/pixel, and a zoom level of 19 corresponds to 0.3 m/pixel. For more information on zoom levels and their corresponding scale and resolution, please refer to the BingMaps API documentation at https://learn.microsoft.com/en-us/bingmaps/articles/understanding-scale-and-resolution

- **model_type**: SAM's image encoder backbone architecture, among 'vit_h', 'vit_l', or 'vit_b'. Before running the workflow, make sure the desired model has been exported to the cluster by running `scripts/export_sam_models.py`. For more information, refer to the FarmVibes.AI troubleshooting page in the documentation.

- **spatial_overlap**: Percentage of spatial overlap between chips in the range of [0.0, 1.0).

## Tasks

- **basemap_download**: Downloads Bing Maps basemap tiles and merges them into a single raster.

- **basemap_segmentation**: Runs Segment Anything Model (SAM) over BingMaps basemap rasters with points and/or bounding boxes as prompts.

## Workflow Yaml

```yaml

name: segment_basemap
sources:
  user_input:
  - basemap_download.input_geometry
  - basemap_segmentation.input_geometry
  prompts:
  - basemap_segmentation.input_prompts
sinks:
  basemap: basemap_download.merged_basemap
  segmentation_mask: basemap_segmentation.segmentation_mask
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
  basemap_segmentation:
    workflow: ml/segment_anything/basemap_prompt_segmentation
    parameters:
      model_type: '@from(model_type)'
      spatial_overlap: '@from(spatial_overlap)'
edges:
- origin: basemap_download.merged_basemap
  destination:
  - basemap_segmentation.input_raster
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