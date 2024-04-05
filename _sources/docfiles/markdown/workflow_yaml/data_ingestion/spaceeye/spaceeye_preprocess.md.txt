# data_ingestion/spaceeye/spaceeye_preprocess

Runs the SpaceEye preprocessing pipeline. The workflow fetches both Sentinel-1 and Sentinel-2 tiles that cover the input geometry and time range and preprocesses them. It also computes improved cloud masks using cloud and shadow segmentation models.

```{mermaid}
    graph TD
    inp1>user_input]
    out1>s2_raster]
    out2>s1_raster]
    out3>cloud_mask]
    tsk1{{s2}}
    tsk2{{s1}}
    tsk1{{s2}} -- raster/s2_products --> tsk2{{s1}}
    inp1>user_input] -- user_input --> tsk1{{s2}}
    inp1>user_input] -- user_input --> tsk2{{s1}}
    tsk1{{s2}} -- raster --> out1>s2_raster]
    tsk2{{s1}} -- raster --> out2>s1_raster]
    tsk1{{s2}} -- mask --> out3>cloud_mask]
```

## Sources

- **user_input**: Time range and geometry of interest.

## Sinks

- **s2_raster**: Sentinel-2 rasters.

- **s1_raster**: Sentinel-1 rasters.

- **cloud_mask**: Cloud and cloud shadow mask.

## Parameters

- **min_tile_cover**: Minimum RoI coverage to consider a set of tiles sufficient.

- **max_tiles_per_time**: Maximum number of tiles used to cover the RoI in each date.

- **cloud_thr**: Confidence threshold to assign a pixel as cloud.

- **shadow_thr**: Confidence threshold to assign a pixel as shadow.

- **pc_key**: Optional Planetary Computer API key.

- **s1_timeout**: Maximum time, in seconds, before a band reading operation times out.

- **s2_timeout**: Maximum time, in seconds, before a band reading operation times out.

## Tasks

- **s2**: Downloads and preprocesses Sentinel-2 imagery that covers the input geometry and time range, and computes improved cloud masks using cloud and shadow segmentation models.

- **s1**: Downloads and preprocesses tiles of Sentinel-1 imagery that intersect with the input Sentinel-2 products in the input time range.

## Workflow Yaml

```yaml

name: spaceeye_preprocess_rtc
sources:
  user_input:
  - s2.user_input
  - s1.user_input
sinks:
  s2_raster: s2.raster
  s1_raster: s1.raster
  cloud_mask: s2.mask
parameters:
  min_tile_cover: 0.4
  max_tiles_per_time: null
  cloud_thr: null
  shadow_thr: null
  pc_key: '@SECRET(eywa-secrets, pc-sub-key)'
  s1_timeout: null
  s2_timeout: null
tasks:
  s2:
    workflow: data_ingestion/sentinel2/preprocess_s2_improved_masks
    parameters:
      min_tile_cover: '@from(min_tile_cover)'
      max_tiles_per_time: '@from(max_tiles_per_time)'
      cloud_thr: '@from(cloud_thr)'
      shadow_thr: '@from(shadow_thr)'
      pc_key: '@from(pc_key)'
      in_memory: true
      dl_timeout: '@from(s2_timeout)'
  s1:
    workflow: data_ingestion/sentinel1/preprocess_s1
    parameters:
      pc_key: '@from(pc_key)'
      dl_timeout: '@from(s1_timeout)'
edges:
- origin: s2.raster
  destination:
  - s1.s2_products
description:
  short_description: Runs the SpaceEye preprocessing pipeline.
  long_description: The workflow fetches both Sentinel-1 and Sentinel-2 tiles that
    cover the input geometry and time range and preprocesses them. It also computes
    improved cloud masks using cloud and shadow segmentation models.
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    s2_raster: Sentinel-2 rasters.
    s1_raster: Sentinel-1 rasters.
    cloud_mask: Cloud and cloud shadow mask.


```