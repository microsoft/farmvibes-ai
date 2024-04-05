# data_ingestion/user_data/ingest_raster

Adds user rasters into the cluster storage, allowing for them to be used on workflows. The workflow downloads rasters provided in the references and generates Raster objects with local assets that can be used in other operations.

```{mermaid}
    graph TD
    inp1>user_input]
    out1>raster]
    tsk1{{unpack}}
    tsk2{{download}}
    tsk1{{unpack}} -- ref_list/input_ref --> tsk2{{download}}
    inp1>user_input] -- input_refs --> tsk1{{unpack}}
    tsk2{{download}} -- downloaded --> out1>raster]
```

## Sources

- **user_input**: List of external references.

## Sinks

- **raster**: Rasters with downloaded assets.

## Tasks

- **unpack**: Unpacks the urls from the list of external references.

- **download**: Downloads the raster from the input reference's url.

## Workflow Yaml

```yaml

name: ingest_raster
sources:
  user_input:
  - unpack.input_refs
sinks:
  raster: download.downloaded
tasks:
  unpack:
    op: unpack_refs
  download:
    op: download_raster_from_ref
    op_dir: download_from_ref
edges:
- origin: unpack.ref_list
  destination:
  - download.input_ref
description:
  short_description: Adds user rasters into the cluster storage, allowing for them
    to be used on workflows.
  long_description: The workflow downloads rasters provided in the references and
    generates Raster objects with local assets that can be used in other operations.
  sources:
    user_input: List of external references.
  sinks:
    raster: Rasters with downloaded assets.


```