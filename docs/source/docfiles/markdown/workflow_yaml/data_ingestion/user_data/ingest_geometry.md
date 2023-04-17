# data_ingestion/user_data/ingest_geometry

```yaml

name: ingest_geometry
sources:
  user_input:
  - unpack.input_refs
sinks:
  geometry: download.downloaded
tasks:
  unpack:
    op: unpack_refs
  download:
    op: download_geometry_from_ref
    op_dir: download_from_ref
edges:
- origin: unpack.ref_list
  destination:
  - download.input_ref
description:
  short_description: Adds user geometries into the cluster storage, allowing for them
    to be used on workflows.
  long_description: The workflow downloads geometries provided in the references and
    generates GeometryCollection objects with local assets that can be used in other
    operations.
  sources:
    user_input: List of external references.
  sinks:
    geometry: GeometryCollections with downloaded assets.


```

```{mermaid}
    graph TD
    inp1>user_input]
    out1>geometry]
    tsk1{{unpack}}
    tsk2{{download}}
    tsk1{{unpack}} -- ref_list/input_ref --> tsk2{{download}}
    inp1>user_input] -- input_refs --> tsk1{{unpack}}
    tsk2{{download}} -- downloaded --> out1>geometry]
```