# data_processing/linear_trend/chunked_linear_trend

Computes the pixel-wise linear trend of a list of rasters (e.g. NDVI). The workflow computes the linear trend over chunks of data, combining them into the final raster.

```{mermaid}
    graph TD
    inp1>input_rasters]
    out1>linear_trend_raster]
    tsk1{{chunk_raster}}
    tsk2{{linear_trend}}
    tsk3{{combine_chunks}}
    tsk1{{chunk_raster}} -- chunk_series/series --> tsk2{{linear_trend}}
    tsk2{{linear_trend}} -- trend/chunks --> tsk3{{combine_chunks}}
    inp1>input_rasters] -- rasters --> tsk1{{chunk_raster}}
    inp1>input_rasters] -- rasters --> tsk2{{linear_trend}}
    tsk3{{combine_chunks}} -- raster --> out1>linear_trend_raster]
```

## Sources

- **input_rasters**: List of rasters to compute linear trend.

## Sinks

- **linear_trend_raster**: Raster with the trend and the test statistics.

## Parameters

- **chunk_step_y**: steps used to divide the rasters into chunks in the y direction (units are grid points).

- **chunk_step_x**: steps used to divide the rasters into chunks in the x direction (units are grid points).

## Tasks

- **chunk_raster**: Splits input rasters into a series of chunks.

- **linear_trend**: Computes the pixel-wise linear trend across rasters.

- **combine_chunks**: Combines series of chunks into a final raster.

## Workflow Yaml

```yaml

name: chunked_linear_trend
sources:
  input_rasters:
  - chunk_raster.rasters
  - linear_trend.rasters
sinks:
  linear_trend_raster: combine_chunks.raster
parameters:
  chunk_step_y: null
  chunk_step_x: null
tasks:
  chunk_raster:
    op: chunk_raster
    parameters:
      step_y: '@from(chunk_step_y)'
      step_x: '@from(chunk_step_x)'
  linear_trend:
    op: linear_trend
  combine_chunks:
    op: combine_chunks
edges:
- origin: chunk_raster.chunk_series
  destination:
  - linear_trend.series
- origin: linear_trend.trend
  destination:
  - combine_chunks.chunks
description:
  short_description: Computes the pixel-wise linear trend of a list of rasters (e.g.
    NDVI).
  long_description: The workflow computes the linear trend over chunks of data, combining
    them into the final raster.
  sources:
    input_rasters: List of rasters to compute linear trend.
  sinks:
    linear_trend_raster: Raster with the trend and the test statistics.
  parameters:
    chunk_step_y: steps used to divide the rasters into chunks in the y direction
      (units are grid points).
    chunk_step_x: steps used to divide the rasters into chunks in the x direction
      (units are grid points).


```