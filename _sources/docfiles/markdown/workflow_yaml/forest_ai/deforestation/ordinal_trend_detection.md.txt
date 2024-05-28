# forest_ai/deforestation/ordinal_trend_detection

Detects increase/decrease trends in the pixel levels over the user-input geometry and time range. This workflow prepares rasters to perform the Cochran-Armitage trend test over a user-provided geometry and time range. Initially, it recodes the input raster according to the 'from_values' and 'to_values' parameters. For example, if the original raster has values (2, 1, 3, 4, 5) and the default values of 'from_values' and 'to_values' are respectively [1, 2, 3, 4, 5] and [6, 7, 8, 9, 10], the recoded raster will have values (7, 6, 8, 9, 10).  The workflow then clips the user-provided geometries and computes an ordinal raster. It also counts each unique pixel present in the recoded rasters to create a pixel frequency contingency table. This data is used to determine if there is an increasing or decreasing trend in pixel levels.  The Cochran-Armitage test is a non-parametric test used to ascertain this trend. The null hypothesis assumes no trend in pixel levels, while the alternative hypothesis assumes a trend exists. The test returns a p-value and a z-score. If the p-value is less than some significance level, the null hypothesis is rejected in favor of the alternative. A positive z-score indicates an increasing trend, while a negative one indicates a decreasing trend.

```{mermaid}
    graph TD
    inp1>raster]
    inp2>input_geometry]
    out1>recoded_raster]
    out2>trend_test_result]
    out3>clipped_raster]
    tsk1{{recode_raster}}
    tsk2{{clip}}
    tsk3{{compute_pixel_count}}
    tsk4{{trend_test}}
    tsk1{{recode_raster}} -- recoded_raster/raster --> tsk2{{clip}}
    tsk2{{clip}} -- clipped_raster/raster --> tsk3{{compute_pixel_count}}
    tsk3{{compute_pixel_count}} -- pixel_count --> tsk4{{trend_test}}
    inp1>raster] -- raster --> tsk1{{recode_raster}}
    inp2>input_geometry] -- input_geometry --> tsk2{{clip}}
    tsk1{{recode_raster}} -- recoded_raster --> out1>recoded_raster]
    tsk4{{trend_test}} -- ordinal_trend_result --> out2>trend_test_result]
    tsk2{{clip}} -- clipped_raster --> out3>clipped_raster]
```

## Sources

- **raster**: Raster to be processed and tested for trends.

- **input_geometry**: Reference geometry.

## Sinks

- **recoded_raster**: Recoded raster for the user-provided geometry and time range.

- **trend_test_result**: Cochran-armitage test results composed of p-value and z-score.

- **clipped_raster**: Clipped ordinal raster for the user-provided geometry and time range.

## Parameters

- **from_values**: List of values to recode from.

- **to_values**: List of values to recode to.

## Tasks

- **recode_raster**: Recodes values of the input raster.

- **clip**: Performs a clip on an input raster based on a provided reference geometry.

- **compute_pixel_count**: Counts the pixel values in the input raster.

- **trend_test**: Detects increase/decrease trends over a list of Rasters.

## Workflow Yaml

```yaml

name: ordinal_trend_detection
sources:
  raster:
  - recode_raster.raster
  input_geometry:
  - clip.input_geometry
sinks:
  recoded_raster: recode_raster.recoded_raster
  trend_test_result: trend_test.ordinal_trend_result
  clipped_raster: clip.clipped_raster
parameters:
  from_values: []
  to_values: []
tasks:
  recode_raster:
    op: recode_raster
    parameters:
      from_values: '@from(from_values)'
      to_values: '@from(to_values)'
  clip:
    workflow: data_processing/clip/clip
  compute_pixel_count:
    op: compute_pixel_count
  trend_test:
    op: ordinal_trend_test
edges:
- origin: recode_raster.recoded_raster
  destination:
  - clip.raster
- origin: clip.clipped_raster
  destination:
  - compute_pixel_count.raster
- origin: compute_pixel_count.pixel_count
  destination:
  - trend_test.pixel_count
description:
  short_description: Detects increase/decrease trends in the pixel levels over the
    user-input geometry and time range.
  long_description: This workflow prepares rasters to perform the Cochran-Armitage
    trend test over a user-provided geometry and time range. Initially, it recodes
    the input raster according to the 'from_values' and 'to_values' parameters. For
    example, if the original raster has values (2, 1, 3, 4, 5) and the default values
    of 'from_values' and 'to_values' are respectively [1, 2, 3, 4, 5] and [6, 7, 8,
    9, 10], the recoded raster will have values (7, 6, 8, 9, 10).  The workflow then
    clips the user-provided geometries and computes an ordinal raster. It also counts
    each unique pixel present in the recoded rasters to create a pixel frequency contingency
    table. This data is used to determine if there is an increasing or decreasing
    trend in pixel levels.  The Cochran-Armitage test is a non-parametric test used
    to ascertain this trend. The null hypothesis assumes no trend in pixel levels,
    while the alternative hypothesis assumes a trend exists. The test returns a p-value
    and a z-score. If the p-value is less than some significance level, the null hypothesis
    is rejected in favor of the alternative. A positive z-score indicates an increasing
    trend, while a negative one indicates a decreasing trend.
  sources:
    raster: Raster to be processed and tested for trends.
    input_geometry: Reference geometry.
  sinks:
    recoded_raster: Recoded raster for the user-provided geometry and time range.
    trend_test_result: Cochran-armitage test results composed of p-value and z-score.
    clipped_raster: Clipped ordinal raster for the user-provided geometry and time
      range.


```