# data_ingestion/admag/admag_seasonal_field

```yaml

name: admag_seasonal_field
sources:
  admag_input:
  - admag_seasonal_field.admag_input
sinks:
  seasonal_field: admag_seasonal_field.seasonal_field
parameters:
  base_url: null
  client_id: null
  client_secret: null
  authority: null
  default_scope: null
tasks:
  admag_seasonal_field:
    op: admag_seasonal_field
    op_dir: admag
    parameters:
      base_url: '@from(base_url)'
      client_id: '@from(client_id)'
      client_secret: '@from(client_secret)'
      authority: '@from(authority)'
      default_scope: '@from(default_scope)'
description:
  short_description: Generates SeasonalFieldInformation using ADMAg (Microsoft Azure
    Data Manager for Agriculture).
  long_description: The workflow creates a DataVibe subclass SeasonalFieldInformation
    that contains farm-related operations (e.g., fertilization, harvest, tillage,
    planting, crop name).
  sources:
    admag_input: Unique identifiers for ADMAg seasonal field, boundary, and farmer.
  sinks:
    seasonal_field: Crop SeasonalFieldInformation which contains SeasonalFieldInformation
      that contains farm-related operations (e.g., fertilization, harvest, tillage,
      planting, crop name).
  parameters:
    base_url: Azure Data Manager for Agriculture host. Please visit https://aka.ms/farmvibesDMA
      to check how to get these credentials.
    client_id: Azure Data Manager for Agriculture client id. Please visit https://aka.ms/farmvibesDMA
      to check how to get these credentials.
    client_secret: Azure Data Manager for Agriculture client secret. Please visit
      https://aka.ms/farmvibesDMA to check how to get these credentials.
    authority: Azure Data Manager for Agriculture authority. Please visit https://aka.ms/farmvibesDMA
      to check how to get these credentials.
    default_scope: Azure Data Manager for Agriculture default scope. Please visit
      https://aka.ms/farmvibesDMA to check how to get these credentials.


```

```{mermaid}
    graph TD
    inp1>admag_input]
    out1>seasonal_field]
    tsk1{{admag_seasonal_field}}
    inp1>admag_input] -- admag_input --> tsk1{{admag_seasonal_field}}
    tsk1{{admag_seasonal_field}} -- seasonal_field --> out1>seasonal_field]
```