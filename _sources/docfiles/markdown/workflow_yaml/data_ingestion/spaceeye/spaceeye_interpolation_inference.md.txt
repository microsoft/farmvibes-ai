# data_ingestion/spaceeye/spaceeye_interpolation_inference

Performs temporal damped interpolation to generate daily cloud-free images given Sentinel-2 data and cloud masks. The workflow will group input Sentinel-2 and cloud mask rasters into spatio-temporal windows and perform inference of each window. The windows will then be merged into rasters for the RoI. More information about SpaceEye available in the paper: https://arxiv.org/abs/2106.08408.

```{mermaid}
    graph TD
    inp1>input_data]
    inp2>s2_rasters]
    inp3>cloud_rasters]
    out1>raster]
    tsk1{{group_s2}}
    tsk2{{group_mask}}
    tsk3{{spaceeye}}
    tsk4{{split}}
    tsk1{{group_s2}} -- tile_sequences/s2_products --> tsk3{{spaceeye}}
    tsk2{{group_mask}} -- tile_sequences/cloud_masks --> tsk3{{spaceeye}}
    tsk3{{spaceeye}} -- spaceeye_sequence/sequences --> tsk4{{split}}
    inp1>input_data] -- input_data --> tsk1{{group_s2}}
    inp1>input_data] -- input_data --> tsk2{{group_mask}}
    inp2>s2_rasters] -- rasters --> tsk1{{group_s2}}
    inp3>cloud_rasters] -- rasters --> tsk2{{group_mask}}
    tsk4{{split}} -- rasters --> out1>raster]
```

## Sources

- **input_data**: Time range and region of interest. Will determine the spatio-temporal windows and region for the output rasters.

- **s2_rasters**: Sentinel-2 tile rasters for the input time range.

- **cloud_rasters**: Cloud masks for each of the Sentinel-2 tiles.

## Sinks

- **raster**: Cloud-free rasters for the input time range and region of interest.

## Parameters

- **duration**: Time window, in days, considered in the inference. Controls the amount of temporal context for inpainting clouds. Larger windows require more compute and memory.

- **time_overlap**: Overlap ratio of each temporal window. Controls the temporal step between windows as a fraction of the window size.

## Tasks

- **group_s2**: Groups Sentinel-2 tiles into time windows of defined duration.

- **group_mask**: Groups Sentinel-2 cloud masks into time windows of defined duration.

- **spaceeye**: Runs the interpolation version of SpaceEye to remove clouds in input rasters.

- **split**: Splits a list of multiple TileSequence back to a list of Rasters.

## Workflow Yaml

```yaml

name: spaceeye_interpolation_inference
sources:
  input_data:
  - group_s2.input_data
  - group_mask.input_data
  s2_rasters:
  - group_s2.rasters
  cloud_rasters:
  - group_mask.rasters
sinks:
  raster: split.rasters
parameters:
  duration: 48
  time_overlap: 0.5
tasks:
  group_s2:
    op: group_s2_tile_sequence
    op_dir: group_tile_sequence
    parameters:
      duration: '@from(duration)'
      overlap: '@from(time_overlap)'
  group_mask:
    op: group_s2cloudmask_tile_sequence
    op_dir: group_tile_sequence
    parameters:
      duration: '@from(duration)'
      overlap: '@from(time_overlap)'
  spaceeye:
    op: remove_clouds_interpolation
    op_dir: remove_clouds
    parameters:
      duration: '@from(duration)'
  split:
    op: split_spaceeye_sequence
    op_dir: split_sequence
edges:
- origin: group_s2.tile_sequences
  destination:
  - spaceeye.s2_products
- origin: group_mask.tile_sequences
  destination:
  - spaceeye.cloud_masks
- origin: spaceeye.spaceeye_sequence
  destination:
  - split.sequences
description:
  short_description: Performs temporal damped interpolation to generate daily cloud-free
    images given Sentinel-2 data and cloud masks.
  long_description: 'The workflow will group input Sentinel-2 and cloud mask rasters
    into spatio-temporal windows and perform inference of each window. The windows
    will then be merged into rasters for the RoI. More information about SpaceEye
    available in the paper: https://arxiv.org/abs/2106.08408.'
  sources:
    input_data: Time range and region of interest. Will determine the spatio-temporal
      windows and region for the output rasters.
    s2_rasters: Sentinel-2 tile rasters for the input time range.
    cloud_rasters: Cloud masks for each of the Sentinel-2 tiles.
  sinks:
    raster: Cloud-free rasters for the input time range and region of interest.
  parameters:
    duration: Time window, in days, considered in the inference. Controls the amount
      of temporal context for inpainting clouds. Larger windows require more compute
      and memory.
    time_overlap: Overlap ratio of each temporal window. Controls the temporal step
      between windows as a fraction of the window size.


```