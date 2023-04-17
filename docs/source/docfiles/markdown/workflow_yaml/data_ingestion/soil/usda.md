# data_ingestion/soil/usda

```yaml

name: usda_soils
sources:
  input_item:
  - datavibe_filter.input_item
sinks:
  downloaded_raster: download_usda_soils.downloaded_raster
parameters:
  ignore: all
tasks:
  datavibe_filter:
    op: datavibe_filter
    parameters:
      filter_out: '@from(ignore)'
  download_usda_soils:
    op: download_usda_soils
edges:
- origin: datavibe_filter.output_item
  destination:
  - download_usda_soils.input_item
description:
  short_description: Downloads USDA soil classification raster.
  long_description: The workflow will download a global raster with USDA soil classes
    at 1/30 degree resolution.
  sources:
    input_item: Dummy input.
  sinks:
    downloaded_raster: Raster with USDA soil classes.
  parameters:
    ignore: Selection of each field of input item should be ignored (among "time_range",
      "geometry", or "all" for both of them).


```

```{mermaid}
    graph TD
    inp1>input_item]
    out1>downloaded_raster]
    tsk1{{datavibe_filter}}
    tsk2{{download_usda_soils}}
    tsk1{{datavibe_filter}} -- output_item/input_item --> tsk2{{download_usda_soils}}
    inp1>input_item] -- input_item --> tsk1{{datavibe_filter}}
    tsk2{{download_usda_soils}} -- downloaded_raster --> out1>downloaded_raster]
```