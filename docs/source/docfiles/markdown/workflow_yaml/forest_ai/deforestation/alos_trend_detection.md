# forest_ai/deforestation/alos_trend_detection

Detects increase/decrease trends in forest pixel levels over the user-input geometry and time range for the ALOS forest map. This workflow combines the alos_forest_extent_download_merge and ordinal_trend_detection workflows to detect increase/decrease trends in the forest pixel levels over the user-provided geometry and time range for the ALOS forest map. The ALOS PALSAR 2.1 Forest/Non-Forest Maps are downloaded in the alos_forest_extent_download_merge workflow.  Then the ordinal_trend_detection workflow clips the ordinal raster to the user-provided geometry and time range and determines if there is an increasing or decreasing trend in the forest pixel levels over them. alos_trend_detection uses the Cochran-Armitage test to detect trends in the forest levels over the years.  The null hypothesis is that there is no trend in the pixel levels over the list of rasters. The alternative hypothesis is that there is a trend in the forest pixel levels over the list of rasters (one for each year). It returns a p-value and a z-score. If the p-value is less than some significance level, the null hypothesis is rejected and the alternative hypothesis is accepted. If the z-score is positive, the trend is increasing.  If the z-score is negative, the trend is decreasing.

```{mermaid}
    graph TD
    inp1>user_input]
    out1>merged_raster]
    out2>categorical_raster]
    out3>recoded_raster]
    out4>clipped_raster]
    out5>trend_test_result]
    tsk1{{alos_forest_extent_download_merge}}
    tsk2{{ordinal_trend_detection}}
    tsk1{{alos_forest_extent_download_merge}} -- merged_raster/raster --> tsk2{{ordinal_trend_detection}}
    inp1>user_input] -- user_input --> tsk1{{alos_forest_extent_download_merge}}
    inp1>user_input] -- input_geometry --> tsk2{{ordinal_trend_detection}}
    tsk1{{alos_forest_extent_download_merge}} -- merged_raster --> out1>merged_raster]
    tsk1{{alos_forest_extent_download_merge}} -- categorical_raster --> out2>categorical_raster]
    tsk2{{ordinal_trend_detection}} -- recoded_raster --> out3>recoded_raster]
    tsk2{{ordinal_trend_detection}} -- clipped_raster --> out4>clipped_raster]
    tsk2{{ordinal_trend_detection}} -- trend_test_result --> out5>trend_test_result]
```

## Sources

- **user_input**: Time range and geometry of interest.

## Sinks

- **merged_raster**: Merged raster of the ALOS PALSAR 2.1 Forest/Non-Forest Map for the user-provided geometry and time range.

- **categorical_raster**: Categorical raster of the ALOS PALSAR 2.1 Forest/Non-Forest Map for the user-provided geometry and time range before the merge operation.

- **recoded_raster**: Recoded raster of the ALOS PALSAR 2.1 Forest/Non-Forest Map for the user-provided geometry and time range.

- **clipped_raster**: Clipped ordinal raster for the user-provided geometry and time range.

- **trend_test_result**: Cochran-armitage test results composed of p-value and z-score.

## Parameters

- **pc_key**: Planetary Computer API key.

- **from_values**: Values to recode from.

- **to_values**: Values to recode to.

## Tasks

- **alos_forest_extent_download_merge**: Downloads Advanced Land Observing Satellite (ALOS) forest/non-forest classification map and merges it into a single raster.

- **ordinal_trend_detection**: Detects increase/decrease trends in the pixel levels over the user-input geometry and time range.

## Workflow Yaml

```yaml

name: alos_trend_detection
sources:
  user_input:
  - alos_forest_extent_download_merge.user_input
  - ordinal_trend_detection.input_geometry
sinks:
  merged_raster: alos_forest_extent_download_merge.merged_raster
  categorical_raster: alos_forest_extent_download_merge.categorical_raster
  recoded_raster: ordinal_trend_detection.recoded_raster
  clipped_raster: ordinal_trend_detection.clipped_raster
  trend_test_result: ordinal_trend_detection.trend_test_result
parameters:
  pc_key: null
  from_values:
  - 4
  - 3
  - 0
  - 2
  - 1
  to_values:
  - 0
  - 0
  - 0
  - 1
  - 1
tasks:
  alos_forest_extent_download_merge:
    workflow: data_ingestion/alos/alos_forest_extent_download_merge
    parameters:
      pc_key: '@from(pc_key)'
  ordinal_trend_detection:
    workflow: forest_ai/deforestation/ordinal_trend_detection
    parameters:
      from_values: '@from(from_values)'
      to_values: '@from(to_values)'
edges:
- origin: alos_forest_extent_download_merge.merged_raster
  destination:
  - ordinal_trend_detection.raster
description:
  short_description: Detects increase/decrease trends in forest pixel levels over
    the user-input geometry and time range for the ALOS forest map.
  long_description: This workflow combines the alos_forest_extent_download_merge and
    ordinal_trend_detection workflows to detect increase/decrease trends in the forest
    pixel levels over the user-provided geometry and time range for the ALOS forest
    map. The ALOS PALSAR 2.1 Forest/Non-Forest Maps are downloaded in the alos_forest_extent_download_merge
    workflow.  Then the ordinal_trend_detection workflow clips the ordinal raster
    to the user-provided geometry and time range and determines if there is an increasing
    or decreasing trend in the forest pixel levels over them. alos_trend_detection
    uses the Cochran-Armitage test to detect trends in the forest levels over the
    years.  The null hypothesis is that there is no trend in the pixel levels over
    the list of rasters. The alternative hypothesis is that there is a trend in the
    forest pixel levels over the list of rasters (one for each year). It returns a
    p-value and a z-score. If the p-value is less than some significance level, the
    null hypothesis is rejected and the alternative hypothesis is accepted. If the
    z-score is positive, the trend is increasing.  If the z-score is negative, the
    trend is decreasing.
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    merged_raster: Merged raster of the ALOS PALSAR 2.1 Forest/Non-Forest Map for
      the user-provided geometry and time range.
    categorical_raster: Categorical raster of the ALOS PALSAR 2.1 Forest/Non-Forest
      Map for the user-provided geometry and time range before the merge operation.
    recoded_raster: Recoded raster of the ALOS PALSAR 2.1 Forest/Non-Forest Map for
      the user-provided geometry and time range.
    clipped_raster: Clipped ordinal raster for the user-provided geometry and time
      range.
    trend_test_result: Cochran-armitage test results composed of p-value and z-score.
  parameters:
    pc_key: Planetary Computer API key.
    from_values: Values to recode from.
    to_values: Values to recode to.


```