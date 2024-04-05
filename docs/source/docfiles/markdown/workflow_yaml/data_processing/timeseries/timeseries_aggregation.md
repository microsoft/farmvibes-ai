# data_processing/timeseries/timeseries_aggregation

Computes the mean, standard deviation, maximum, and minimum values of all regions of the raster and aggregates them into a timeseries. 

```{mermaid}
    graph TD
    inp1>raster]
    inp2>input_geometry]
    out1>timeseries]
    tsk1{{summary}}
    tsk2{{timeseries}}
    tsk1{{summary}} -- summary/stats --> tsk2{{timeseries}}
    inp1>raster] -- raster --> tsk1{{summary}}
    inp2>input_geometry] -- input_geometry --> tsk1{{summary}}
    tsk2{{timeseries}} -- timeseries --> out1>timeseries]
```

## Sources

- **raster**: Input raster.

- **input_geometry**: Geometry of interest.

## Sinks

- **timeseries**: Aggregated statistics of the raster.

## Tasks

- **summary**: Computes the mean, standard deviation, maximum, and minimum values across the whole raster.

- **timeseries**: Aggregates list of summary statistics into a timeseries.

## Workflow Yaml

```yaml

name: timeseries_aggregation
sources:
  raster:
  - summary.raster
  input_geometry:
  - summary.input_geometry
sinks:
  timeseries: timeseries.timeseries
tasks:
  summary:
    op: summarize_raster
  timeseries:
    op: aggregate_statistics_timeseries
edges:
- origin: summary.summary
  destination:
  - timeseries.stats
description:
  short_description: Computes the mean, standard deviation, maximum, and minimum values
    of all regions of the raster and aggregates them into a timeseries.
  long_description: null
  sources:
    raster: Input raster.
    input_geometry: Geometry of interest.
  sinks:
    timeseries: Aggregated statistics of the raster.


```