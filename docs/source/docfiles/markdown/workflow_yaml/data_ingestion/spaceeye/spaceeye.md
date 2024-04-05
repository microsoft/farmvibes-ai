# data_ingestion/spaceeye/spaceeye

Runs the SpaceEye cloud removal pipeline, yielding daily cloud-free images for the input geometry and time range. The workflow fetches both Sentinel-1 and Sentinel-2 tiles that cover the input geometry and time range, preprocesses them, computes cloud masks, and runs SpaceEye inference in a sliding window on the retrieved tiles. This workflow can be reused as a preprocess step in many applications that require cloud-free Sentinel-2 data. For more information about SpaceEye, read the paper: https://arxiv.org/abs/2106.08408.

```{mermaid}
    graph TD
    inp1>user_input]
    out1>raster]
    tsk1{{preprocess}}
    tsk2{{spaceeye}}
    tsk1{{preprocess}} -- s2_raster/s2_rasters --> tsk2{{spaceeye}}
    tsk1{{preprocess}} -- s1_raster/s1_rasters --> tsk2{{spaceeye}}
    tsk1{{preprocess}} -- cloud_mask/cloud_rasters --> tsk2{{spaceeye}}
    inp1>user_input] -- user_input --> tsk1{{preprocess}}
    inp1>user_input] -- input_data --> tsk2{{spaceeye}}
    tsk2{{spaceeye}} -- raster --> out1>raster]
```

## Sources

- **user_input**: Time range and geometry of interest.

## Sinks

- **raster**: Cloud-free rasters.

## Parameters

- **duration**: Time window, in days, considered in the inference. Controls the amount of temporal context for inpainting clouds. Larger windows require more compute and memory.

- **time_overlap**: Overlap ratio of each temporal window. Controls the temporal step between windows as a fraction of the window size.

- **min_tile_cover**: Minimum RoI coverage to consider a set of tiles sufficient.

- **max_tiles_per_time**: Maximum number of tiles used to cover the RoI in each date.

- **cloud_thr**: Confidence threshold to assign a pixel as cloud.

- **shadow_thr**: Confidence threshold to assign a pixel as shadow.

- **pc_key**: Optional Planetary Computer API key.

- **s2_timeout**: Maximum time, in seconds, before a band reading operation times out.

## Tasks

- **preprocess**: Runs the SpaceEye preprocessing pipeline.

- **spaceeye**: Performs SpaceEye inference to generate daily cloud-free images given Sentinel data and cloud masks.

## Workflow Yaml

```yaml

name: spaceeye
sources:
  user_input:
  - preprocess.user_input
  - spaceeye.input_data
sinks:
  raster: spaceeye.raster
parameters:
  duration: null
  time_overlap: null
  min_tile_cover: null
  max_tiles_per_time: null
  cloud_thr: null
  shadow_thr: null
  pc_key: null
  s2_timeout: null
tasks:
  preprocess:
    workflow: data_ingestion/spaceeye/spaceeye_preprocess
    parameters:
      min_tile_cover: '@from(min_tile_cover)'
      max_tiles_per_time: '@from(max_tiles_per_time)'
      cloud_thr: '@from(cloud_thr)'
      shadow_thr: '@from(shadow_thr)'
      pc_key: '@from(pc_key)'
      s2_timeout: '@from(s2_timeout)'
  spaceeye:
    workflow: data_ingestion/spaceeye/spaceeye_inference
    parameters:
      duration: '@from(duration)'
      time_overlap: '@from(time_overlap)'
edges:
- origin: preprocess.s2_raster
  destination:
  - spaceeye.s2_rasters
- origin: preprocess.s1_raster
  destination:
  - spaceeye.s1_rasters
- origin: preprocess.cloud_mask
  destination:
  - spaceeye.cloud_rasters
description:
  short_description: Runs the SpaceEye cloud removal pipeline, yielding daily cloud-free
    images for the input geometry and time range.
  long_description: 'The workflow fetches both Sentinel-1 and Sentinel-2 tiles that
    cover the input geometry and time range, preprocesses them, computes cloud masks,
    and runs SpaceEye inference in a sliding window on the retrieved tiles. This workflow
    can be reused as a preprocess step in many applications that require cloud-free
    Sentinel-2 data. For more information about SpaceEye, read the paper: https://arxiv.org/abs/2106.08408.'
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    raster: Cloud-free rasters.
  parameters: null


```