# data_processing/threshold/threshold_raster

Thresholds values of the input raster if higher than the threshold parameter. 

```{mermaid}
    graph TD
    inp1>raster]
    out1>thresholded_raster]
    tsk1{{threshold_task}}
    inp1>raster] -- raster --> tsk1{{threshold_task}}
    tsk1{{threshold_task}} -- thresholded --> out1>thresholded_raster]
```

## Sources

- **raster**: Input raster.

## Sinks

- **thresholded_raster**: Thresholded raster.

## Parameters

- **threshold**: Threshold value.

## Tasks

- **threshold_task**: Thresholds values of the input raster if higher than the threshold parameter.

## Workflow Yaml

```yaml

name: threshold_raster
sources:
  raster:
  - threshold_task.raster
sinks:
  thresholded_raster: threshold_task.thresholded
parameters:
  threshold: null
tasks:
  threshold_task:
    op: threshold_raster
    parameters:
      threshold: '@from(threshold)'
edges: null
description:
  short_description: Thresholds values of the input raster if higher than the threshold
    parameter.
  long_description: null
  sources:
    raster: Input raster.
  sinks:
    thresholded_raster: Thresholded raster.
  parameters:
    threshold: Threshold value.


```