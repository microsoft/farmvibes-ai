# farm_ai/agriculture/ndvi_summary

```yaml

name: ndvi_summary
sources:
  user_input:
  - s2.user_input
  - summary_timeseries.input_geometry
sinks:
  timeseries: summary_timeseries.timeseries
parameters:
  pc_key: null
tasks:
  s2:
    workflow: data_ingestion/sentinel2/preprocess_s2_improved_masks
    parameters:
      max_tiles_per_time: 1
      pc_key: '@from(pc_key)'
  compute_ndvi:
    workflow: data_processing/index/index
  summary_timeseries:
    workflow: data_processing/timeseries/timeseries_masked_aggregation
edges:
- origin: s2.raster
  destination:
  - compute_ndvi.raster
- origin: compute_ndvi.index_raster
  destination:
  - summary_timeseries.raster
- origin: s2.mask
  destination:
  - summary_timeseries.mask
description:
  short_description: Calculates NDVI statistics (mean, standard deviation, maximum
    and minimum) for the input geometry and time range.
  long_description: The workflow retrieves the relevant Sentinel-2 products with Planetary
    Computer (PC) API, forwards them to a cloud detection model and combines the predicted
    cloud mask to the mask obtained from the product. The workflow computes the NDVI
    for each available tile and date, summarizing each with the mean, standard deviation,
    maximum and minimum values for the regions not obscured by clouds. Finally, it
    outputs a timeseries with such statistics for all available dates, ignoring heavily-clouded
    tiles.
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    timeseries: Aggregated NDVI statistics of the retrieved tiles within the input
      geometry and time range.
  parameters:
    pc_key: Optional Planetary Computer API key.


```

```{mermaid}
    graph TD
    inp1>user_input]
    out1>timeseries]
    tsk1{{s2}}
    tsk2{{compute_ndvi}}
    tsk3{{summary_timeseries}}
    tsk1{{s2}} -- raster --> tsk2{{compute_ndvi}}
    tsk2{{compute_ndvi}} -- index_raster/raster --> tsk3{{summary_timeseries}}
    tsk1{{s2}} -- mask --> tsk3{{summary_timeseries}}
    inp1>user_input] -- user_input --> tsk1{{s2}}
    inp1>user_input] -- input_geometry --> tsk3{{summary_timeseries}}
    tsk3{{summary_timeseries}} -- timeseries --> out1>timeseries]
```