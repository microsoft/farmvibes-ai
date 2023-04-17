# farm_ai/land_degradation/ndvi_linear_trend

```yaml

name: ndvi_linear_trend
sources:
  raster:
  - ndvi.raster
sinks:
  ndvi_raster: ndvi.index_raster
  linear_trend: chunked_linear_trend.linear_trend_raster
tasks:
  ndvi:
    workflow: data_processing/index/index
    parameters:
      index: ndvi
  chunked_linear_trend:
    workflow: data_processing/linear_trend/chunked_linear_trend
    parameters:
      chunk_step_y: 512
      chunk_step_x: 512
edges:
- origin: ndvi.index_raster
  destination:
  - chunked_linear_trend.input_rasters
description:
  short_description: Computes the pixel-wise NDVI linear trend over the input raster.
  long_description: The workflow computes the NDVI from the input raster, calculates
    the linear trend over chunks of data, combining them into the final raster.
  sources:
    raster: Input raster.
  sinks:
    ndvi_raster: NDVI raster.
    linear_trend: Raster with the trend and the test statistics.


```

```{mermaid}
    graph TD
    inp1>raster]
    out1>ndvi_raster]
    out2>linear_trend]
    tsk1{{ndvi}}
    tsk2{{chunked_linear_trend}}
    tsk1{{ndvi}} -- index_raster/input_rasters --> tsk2{{chunked_linear_trend}}
    inp1>raster] -- raster --> tsk1{{ndvi}}
    tsk1{{ndvi}} -- index_raster --> out1>ndvi_raster]
    tsk2{{chunked_linear_trend}} -- linear_trend_raster --> out2>linear_trend]
```