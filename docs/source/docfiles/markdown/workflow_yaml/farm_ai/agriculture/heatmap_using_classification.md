# farm_ai/agriculture/heatmap_using_classification

```yaml

name: heatmap_using_classification
sources:
  input_samples:
  - download_samples.user_input
  input_raster:
  - soil_sample_heatmap_classification.input_raster
sinks:
  result: soil_sample_heatmap_classification.result
parameters:
  attribute_name: null
  buffer: null
  index: null
  bins: null
  simplify: null
  tolerance: null
  data_scale: null
  distribute_output: null
  max_depth: null
  n_estimators: null
  random_state: null
tasks:
  download_samples:
    workflow: data_ingestion/user_data/ingest_geometry
  soil_sample_heatmap_classification:
    workflow: data_processing/heatmap/classification
    parameters:
      attribute_name: '@from(attribute_name)'
      buffer: '@from(buffer)'
      index: '@from(index)'
      bins: '@from(bins)'
      simplify: '@from(simplify)'
      tolerance: '@from(tolerance)'
      data_scale: '@from(data_scale)'
      distribute_output: '@from(distribute_output)'
      max_depth: '@from(max_depth)'
      n_estimators: '@from(n_estimators)'
      random_state: '@from(random_state)'
edges:
- origin: download_samples.geometry
  destination:
  - soil_sample_heatmap_classification.samples
description:
  short_description: The workflow generates a nutrient heatmap for samples provided
    by user by downloading the samples from user input.
  long_description: The samples provided are related with farm boundary and have required
    nutrient information to create a heatmap.
  sources:
    input_raster: Input raster for index computation.
    input_samples: External references to sensor samples for nutrients.
  sinks:
    result: Zip file containing cluster geometries.
  parameters: null


```

```{mermaid}
    graph TD
    inp1>input_samples]
    inp2>input_raster]
    out1>result]
    tsk1{{download_samples}}
    tsk2{{soil_sample_heatmap_classification}}
    tsk1{{download_samples}} -- geometry/samples --> tsk2{{soil_sample_heatmap_classification}}
    inp1>input_samples] -- user_input --> tsk1{{download_samples}}
    inp2>input_raster] -- input_raster --> tsk2{{soil_sample_heatmap_classification}}
    tsk2{{soil_sample_heatmap_classification}} -- result --> out1>result]
```