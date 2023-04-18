# farm_ai/land_degradation/landsat_ndvi_trend

```yaml

name: landsat_ndvi_trend
sources:
  user_input:
  - landsat.user_input
sinks:
  ndvi: trend.ndvi_raster
  linear_trend: trend.linear_trend
parameters:
  pc_key: null
tasks:
  landsat:
    workflow: data_ingestion/landsat/preprocess_landsat
    parameters:
      pc_key: '@from(pc_key)'
  trend:
    workflow: farm_ai/land_degradation/ndvi_linear_trend
edges:
- origin: landsat.raster
  destination:
  - trend.raster
description:
  short_description: Estimates a linear trend over NDVI computer over LANDSAT tiles
    that intersect with the input geometry and time range.
  long_description: The workflow downloads LANDSAT data, compute NDVI over them, and
    estimate a linear trend over chunks of data, combining them into a final trend
    raster.
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    ndvi: NDVI rasters.
    linear_trend: Raster with the trend and the test statistics.
  parameters:
    pc_key: Optional Planetary Computer API key.


```

```{mermaid}
    graph TD
    inp1>user_input]
    out1>ndvi]
    out2>linear_trend]
    tsk1{{landsat}}
    tsk2{{trend}}
    tsk1{{landsat}} -- raster --> tsk2{{trend}}
    inp1>user_input] -- user_input --> tsk1{{landsat}}
    tsk2{{trend}} -- ndvi_raster --> out1>ndvi]
    tsk2{{trend}} -- linear_trend --> out2>linear_trend]
```