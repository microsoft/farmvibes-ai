# data_ingestion/sentinel2/preprocess_s2_improved_masks

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