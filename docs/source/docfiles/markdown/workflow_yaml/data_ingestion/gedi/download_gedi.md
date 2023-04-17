# data_ingestion/gedi/download_gedi

```yaml

name: download_gedi
sources:
  user_input:
  - list.input_data
sinks:
  product: download.downloaded_product
parameters:
  earthdata_token: null
  processing_level: null
tasks:
  list:
    op: list_gedi_products
    parameters:
      processing_level: '@from(processing_level)'
  download:
    op: download_gedi_product
    parameters:
      token: '@from(earthdata_token)'
edges:
- origin: list.gedi_products
  destination:
  - download.gedi_product
description:
  short_description: Downloads GEDI products for the input region and time range.
  long_description: The workflow downloads Global Ecosystem Dynamics Investigation
    (GEDI) products at the desired processing level using NASA's EarthData API. This
    workflow requires an EarthData API token.
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    product: GEDI products.
  parameters:
    earthdata_token: API token for the EarthData platform. Required to run the workflow.
    processing_level: GEDI product processing level. One of 'GEDI01_B.002', 'GEDI02_A.002',
      'GEDI02_B.002'.


```

```{mermaid}
    graph TD
    inp1>user_input]
    out1>product]
    tsk1{{list}}
    tsk2{{download}}
    tsk1{{list}} -- gedi_products/gedi_product --> tsk2{{download}}
    inp1>user_input] -- input_data --> tsk1{{list}}
    tsk2{{download}} -- downloaded_product --> out1>product]
```