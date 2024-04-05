# farm_ai/agriculture/change_detection

Identifies changes/outliers over NDVI across dates. The workflow generates SpaceEye imagery for the input region and time range and computes NDVI raster for each date. It aggregates NDVI statistics (mean, standard deviation, maximum and minimum) in time and detects outliers across dates with a single-component Gaussian Mixture Model (GMM).

```{mermaid}
    graph TD
    inp1>user_input]
    out1>spaceeye_raster]
    out2>index]
    out3>timeseries]
    out4>segmentation]
    out5>heatmap]
    out6>outliers]
    out7>mixture_means]
    tsk1{{spaceeye}}
    tsk2{{ndvi}}
    tsk3{{summary_timeseries}}
    tsk4{{outliers}}
    tsk1{{spaceeye}} -- raster --> tsk2{{ndvi}}
    tsk2{{ndvi}} -- index_raster/raster --> tsk3{{summary_timeseries}}
    tsk2{{ndvi}} -- index_raster/rasters --> tsk4{{outliers}}
    inp1>user_input] -- user_input --> tsk1{{spaceeye}}
    inp1>user_input] -- input_geometry --> tsk3{{summary_timeseries}}
    tsk1{{spaceeye}} -- raster --> out1>spaceeye_raster]
    tsk2{{ndvi}} -- index_raster --> out2>index]
    tsk3{{summary_timeseries}} -- timeseries --> out3>timeseries]
    tsk4{{outliers}} -- segmentation --> out4>segmentation]
    tsk4{{outliers}} -- heatmap --> out5>heatmap]
    tsk4{{outliers}} -- outliers --> out6>outliers]
    tsk4{{outliers}} -- mixture_means --> out7>mixture_means]
```

## Sources

- **user_input**: Time range and geometry of interest.

## Sinks

- **spaceeye_raster**: SpaceEye cloud-free rasters.

- **index**: NDVI rasters.

- **timeseries**: Aggregated NDVI statistics over the time range.

- **segmentation**: Segmentation maps based on the likelihood of each sample belonging to the GMM's single-component.

- **heatmap**: Likelihood maps.

- **outliers**: Outlier maps.

- **mixture_means**: Means of the GMM.

## Parameters

- **pc_key**: PlanetaryComputer API key.

## Tasks

- **spaceeye**: Runs the SpaceEye cloud removal pipeline, yielding daily cloud-free images for the input geometry and time range.

- **ndvi**: Computes an index from the bands of an input raster.

- **summary_timeseries**: Computes the mean, standard deviation, maximum, and minimum values of all regions of the raster and aggregates them into a timeseries.

- **outliers**: Fits a single-component Gaussian Mixture Model (GMM) over input data to detect outliers according to the threshold parameter.

## Workflow Yaml

```yaml

name: change_detection
sources:
  user_input:
  - spaceeye.user_input
  - summary_timeseries.input_geometry
sinks:
  spaceeye_raster: spaceeye.raster
  index: ndvi.index_raster
  timeseries: summary_timeseries.timeseries
  segmentation: outliers.segmentation
  heatmap: outliers.heatmap
  outliers: outliers.outliers
  mixture_means: outliers.mixture_means
parameters:
  pc_key: null
tasks:
  spaceeye:
    workflow: data_ingestion/spaceeye/spaceeye
    parameters:
      pc_key: '@from(pc_key)'
  ndvi:
    workflow: data_processing/index/index
    parameters:
      index: ndvi
  summary_timeseries:
    workflow: data_processing/timeseries/timeseries_aggregation
  outliers:
    workflow: data_processing/outlier/detect_outlier
edges:
- origin: spaceeye.raster
  destination:
  - ndvi.raster
- origin: ndvi.index_raster
  destination:
  - summary_timeseries.raster
  - outliers.rasters
description:
  short_description: Identifies changes/outliers over NDVI across dates.
  long_description: The workflow generates SpaceEye imagery for the input region and
    time range and computes NDVI raster for each date. It aggregates NDVI statistics
    (mean, standard deviation, maximum and minimum) in time and detects outliers across
    dates with a single-component Gaussian Mixture Model (GMM).
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    spaceeye_raster: SpaceEye cloud-free rasters.
    index: NDVI rasters.
    timeseries: Aggregated NDVI statistics over the time range.
    segmentation: Segmentation maps based on the likelihood of each sample belonging
      to the GMM's single-component.
    heatmap: Likelihood maps.
    outliers: Outlier maps.
    mixture_means: Means of the GMM.
  parameters:
    pc_key: PlanetaryComputer API key.


```