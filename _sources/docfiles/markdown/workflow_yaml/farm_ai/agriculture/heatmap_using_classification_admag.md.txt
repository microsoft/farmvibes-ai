# farm_ai/agriculture/heatmap_using_classification_admag

```yaml

name: heatmap_using_classification_admag
sources:
  admag_input:
  - prescriptions.admag_input
  input_raster:
  - soil_sample_heatmap_classification.input_raster
sinks:
  result: soil_sample_heatmap_classification.result
parameters:
  base_url: null
  client_id: null
  client_secret: null
  authority: null
  default_scope: null
  attribute_name: null
  buffer: null
  index: null
  bins: null
  simplify: null
  tolerance: null
  data_scale: null
  max_depth: null
  n_estimators: null
  random_state: null
tasks:
  prescriptions:
    workflow: data_ingestion/admag/prescriptions
    parameters:
      base_url: '@from(base_url)'
      client_id: '@from(client_id)'
      client_secret: '@from(client_secret)'
      authority: '@from(authority)'
      default_scope: '@from(default_scope)'
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
      max_depth: '@from(max_depth)'
      n_estimators: '@from(n_estimators)'
      random_state: '@from(random_state)'
edges:
- origin: prescriptions.response
  destination:
  - soil_sample_heatmap_classification.samples
description:
  short_description: This workflow integrate the ADMAG API to download prescriptions
    and generate heatmap.
  long_description: The prescriptions are related with farm boundary and the nutrient
    information. Each prescription represent a sensor sample at a location within
    a farm boundary.
  sources:
    input_raster: Input raster for index computation.
    admag_input: Required inputs to download prescriptions from admag.
  sinks:
    result: Zip file containing cluster geometries.
  parameters:
    base_url: URL to access the registered app
    client_id: Value uniquely identifies registered application in the Microsoft identity
      platform. Visit url https://learn.microsoft.com/en-us/azure/active-directory/develop/quickstart-register-app
      to register the app.
    client_secret: Sometimes called an application password, a client secret is a
      string value your app can use in place of a certificate to identity itself.
    authority: The endpoint URIs for your app are generated automatically when you
      register or configure your app. It is used by client to obtain authorization
      from the resource owner
    default_scope: URL for default azure OAuth2 permissions


```

```{mermaid}
    graph TD
    inp1>admag_input]
    inp2>input_raster]
    out1>result]
    tsk1{{prescriptions}}
    tsk2{{soil_sample_heatmap_classification}}
    tsk1{{prescriptions}} -- response/samples --> tsk2{{soil_sample_heatmap_classification}}
    inp1>admag_input] -- admag_input --> tsk1{{prescriptions}}
    inp2>input_raster] -- input_raster --> tsk2{{soil_sample_heatmap_classification}}
    tsk2{{soil_sample_heatmap_classification}} -- result --> out1>result]
```