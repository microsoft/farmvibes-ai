# farm_ai/agriculture/emergence_summary

```yaml

name: emergence_summary
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
  msavi:
    workflow: data_processing/index/index
    parameters:
      index: msavi
  emergence:
    workflow: data_processing/threshold/threshold_raster
    parameters:
      threshold: 0.2
  summary_timeseries:
    workflow: data_processing/timeseries/timeseries_masked_aggregation
edges:
- origin: s2.raster
  destination:
  - msavi.raster
- origin: msavi.index_raster
  destination:
  - emergence.raster
- origin: emergence.thresholded_raster
  destination:
  - summary_timeseries.raster
- origin: s2.mask
  destination:
  - summary_timeseries.mask
description:
  short_description: Calculates emergence statistics using thresholded MSAVI (mean,
    standard deviation, maximum and minimum) for the input geometry and time range.
  long_description: The workflow retrieves Sentinel2 products with Planetary Computer
    (PC) API, forwards them to a cloud detection model and combines the predicted
    cloud mask to the mask provided by PC. It computes the MSAVI for each available
    tile and date, thresholds them above a certain value and summarizes each with
    the mean, standard deviation, maximum and minimum values for the regions not obscured
    by clouds. Finally, it outputs a timeseries with such statistics for all available
    dates, filtering out heavily-clouded tiles.
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    timeseries: Aggregated emergence statistics of the retrieved tiles within the
      input geometry and time range.
  parameters:
    pc_key: Optional Planetary Computer API key.


```

```{mermaid}
    graph TD
    inp1>user_input]
    out1>timeseries]
    tsk1{{s2}}
    tsk2{{msavi}}
    tsk3{{emergence}}
    tsk4{{summary_timeseries}}
    tsk1{{s2}} -- raster --> tsk2{{msavi}}
    tsk2{{msavi}} -- index_raster/raster --> tsk3{{emergence}}
    tsk3{{emergence}} -- thresholded_raster/raster --> tsk4{{summary_timeseries}}
    tsk1{{s2}} -- mask --> tsk4{{summary_timeseries}}
    inp1>user_input] -- user_input --> tsk1{{s2}}
    inp1>user_input] -- input_geometry --> tsk4{{summary_timeseries}}
    tsk4{{summary_timeseries}} -- timeseries --> out1>timeseries]
```