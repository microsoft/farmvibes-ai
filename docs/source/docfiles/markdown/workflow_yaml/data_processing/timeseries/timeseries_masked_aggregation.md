# data_processing/timeseries/timeseries_masked_aggregation

Computes the mean, standard deviation, maximum, and minimum values of all regions of the raster considered by the mask and aggregates them into a timeseries. 

```{mermaid}
    graph TD
    inp1>raster]
    inp2>mask]
    inp3>input_geometry]
    out1>timeseries]
    tsk1{{masked_summary}}
    tsk2{{timeseries}}
    tsk1{{masked_summary}} -- summary/stats --> tsk2{{timeseries}}
    inp1>raster] -- raster --> tsk1{{masked_summary}}
    inp2>mask] -- mask --> tsk1{{masked_summary}}
    inp3>input_geometry] -- input_geometry --> tsk1{{masked_summary}}
    tsk2{{timeseries}} -- timeseries --> out1>timeseries]
```

## Sources

- **raster**: Input raster.

- **mask**: Mask of the regions to be considered during summarization;

- **input_geometry**: Geometry of interest.

## Sinks

- **timeseries**: Aggregated statistics of the raster considered by the mask.

## Parameters

- **timeseries_masked_thr**: Threshold of the maximum ratio of masked content allowed in a raster. The statistics of rasters with masked content above the threshold (e.g., heavily clouded) are not included in the timeseries.

## Tasks

- **masked_summary**: Computes the mean, standard deviation, maximum, and minimum values across non-masked regions of the raster.

- **timeseries**: Aggregates list of summary statistics into a timeseries.

## Workflow Yaml

```yaml

name: timeseries_masked_aggregation
sources:
  raster:
  - masked_summary.raster
  mask:
  - masked_summary.mask
  input_geometry:
  - masked_summary.input_geometry
sinks:
  timeseries: timeseries.timeseries
parameters:
  timeseries_masked_thr: null
tasks:
  masked_summary:
    op: summarize_masked_raster
    op_dir: summarize_raster
  timeseries:
    op: aggregate_statistics_timeseries
    parameters:
      masked_thr: '@from(timeseries_masked_thr)'
edges:
- origin: masked_summary.summary
  destination:
  - timeseries.stats
description:
  short_description: Computes the mean, standard deviation, maximum, and minimum values
    of all regions of the raster considered by the mask and aggregates them into a
    timeseries.
  long_description: null
  sources:
    raster: Input raster.
    mask: Mask of the regions to be considered during summarization;
    input_geometry: Geometry of interest.
  sinks:
    timeseries: Aggregated statistics of the raster considered by the mask.
  parameters:
    timeseries_masked_thr: Threshold of the maximum ratio of masked content allowed
      in a raster. The statistics of rasters with masked content above the threshold
      (e.g., heavily clouded) are not included in the timeseries.


```