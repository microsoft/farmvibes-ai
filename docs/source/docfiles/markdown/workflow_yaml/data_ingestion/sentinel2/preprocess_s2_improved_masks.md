# data_ingestion/sentinel2/preprocess_s2_improved_masks

Downloads and preprocesses Sentinel-2 imagery that covers the input geometry and time range, and computes improved cloud masks using cloud and shadow segmentation models. This workflow selects a minimum set of tiles that covers the input geometry, downloads Sentinel-2 imagery for the selected time range, and preprocesses it by generating a single multi-band raster at 10m resolution. It then improves cloud masks by merging the product mask with cloud and shadow masks computed using cloud and shadow segmentation models.

```{mermaid}
    graph TD
    inp1>user_input]
    out1>raster]
    out2>mask]
    tsk1{{s2}}
    tsk2{{cloud}}
    tsk1{{s2}} -- raster/s2_raster --> tsk2{{cloud}}
    tsk1{{s2}} -- mask/product_mask --> tsk2{{cloud}}
    inp1>user_input] -- user_input --> tsk1{{s2}}
    tsk1{{s2}} -- raster --> out1>raster]
    tsk2{{cloud}} -- mask --> out2>mask]
```

## Sources

- **user_input**: Time range and geometry of interest.

## Sinks

- **raster**: Sentinel-2 L2A rasters with all bands resampled to 10m resolution.

- **mask**: Cloud masks at 10m resolution.

## Parameters

- **min_tile_cover**: Minimum RoI coverage to consider a set of tiles sufficient.

- **max_tiles_per_time**: Maximum number of tiles used to cover the RoI in each date.

- **cloud_thr**: Confidence threshold to assign a pixel as cloud.

- **shadow_thr**: Confidence threshold to assign a pixel as shadow.

- **in_memory**: Whether to load the whole raster in memory when running predictions. Uses more memory (~4GB/worker) but speeds up inference for fast models.

- **cloud_model**: ONNX file for the cloud model. Available models are 'cloud_model{idx}_cpu.onnx' with idx ∈ {1, 2} being FPN-based models, which are more accurate but slower, and idx ∈ {3, 4, 5} being cheaplab models, which are less accurate but faster.

- **shadow_model**: ONNX file for the shadow model. 'shadow.onnx' is the only currently available model.

- **pc_key**: Optional Planetary Computer API key.

- **dl_timeout**: Maximum time, in seconds, before a band reading operation times out.

## Tasks

- **s2**: Downloads and preprocesses Sentinel-2 imagery that covers the input geometry and time range.

- **cloud**: Improves cloud masks by merging the product cloud mask with cloud and shadow masks computed by machine learning segmentation models.

## Workflow Yaml

```yaml

name: preprocess_s2_improved_masks
sources:
  user_input:
  - s2.user_input
sinks:
  raster: s2.raster
  mask: cloud.mask
parameters:
  min_tile_cover: null
  max_tiles_per_time: null
  cloud_thr: null
  shadow_thr: null
  in_memory: null
  cloud_model: null
  shadow_model: null
  pc_key: null
  dl_timeout: null
tasks:
  s2:
    workflow: data_ingestion/sentinel2/preprocess_s2
    parameters:
      min_tile_cover: '@from(min_tile_cover)'
      max_tiles_per_time: '@from(max_tiles_per_time)'
      pc_key: '@from(pc_key)'
      dl_timeout: '@from(dl_timeout)'
  cloud:
    workflow: data_ingestion/sentinel2/improve_cloud_mask
    parameters:
      cloud_thr: '@from(cloud_thr)'
      shadow_thr: '@from(shadow_thr)'
      in_memory: '@from(in_memory)'
      cloud_model: '@from(cloud_model)'
      shadow_model: '@from(shadow_model)'
edges:
- origin: s2.raster
  destination:
  - cloud.s2_raster
- origin: s2.mask
  destination:
  - cloud.product_mask
description:
  short_description: Downloads and preprocesses Sentinel-2 imagery that covers the
    input geometry and time range, and computes improved cloud masks using cloud and
    shadow segmentation models.
  long_description: This workflow selects a minimum set of tiles that covers the input
    geometry, downloads Sentinel-2 imagery for the selected time range, and preprocesses
    it by generating a single multi-band raster at 10m resolution. It then improves
    cloud masks by merging the product mask with cloud and shadow masks computed using
    cloud and shadow segmentation models.
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    raster: Sentinel-2 L2A rasters with all bands resampled to 10m resolution.
    mask: Cloud masks at 10m resolution.


```