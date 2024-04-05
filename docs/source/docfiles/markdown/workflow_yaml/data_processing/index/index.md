# data_processing/index/index

Computes an index from the bands of an input raster. In addition to the indices 'ndvi', 'evi', 'msavi', 'ndre', 'reci', 'ndmi', 'methane' and 'pri' all indices in https://github.com/awesome-spectral-indices/awesome-spectral-indices are available (depending on the bands available on the corresponding satellite product).

```{mermaid}
    graph TD
    inp1>raster]
    out1>index_raster]
    tsk1{{compute_index}}
    inp1>raster] -- raster --> tsk1{{compute_index}}
    tsk1{{compute_index}} -- index --> out1>index_raster]
```

## Sources

- **raster**: Input raster.

## Sinks

- **index_raster**: Single-band raster with the computed index.

## Parameters

- **index**: The choice of index to be computed ('ndvi', 'evi', 'msavi', 'ndre', 'reci', 'ndmi', 'methane', 'pri' or any of the awesome-spectral-indices).

## Tasks

- **compute_index**: Computes `index` over the input raster.

## Workflow Yaml

```yaml

name: index
sources:
  raster:
  - compute_index.raster
sinks:
  index_raster: compute_index.index
parameters:
  index: ndvi
tasks:
  compute_index:
    op: compute_index
    parameters:
      index: '@from(index)'
edges: null
description:
  short_description: Computes an index from the bands of an input raster.
  long_description: In addition to the indices 'ndvi', 'evi', 'msavi', 'ndre', 'reci',
    'ndmi', 'methane' and 'pri' all indices in https://github.com/awesome-spectral-indices/awesome-spectral-indices
    are available (depending on the bands available on the corresponding satellite
    product).
  sources:
    raster: Input raster.
  sinks:
    index_raster: Single-band raster with the computed index.
  parameters:
    index: The choice of index to be computed ('ndvi', 'evi', 'msavi', 'ndre', 'reci',
      'ndmi', 'methane', 'pri' or any of the awesome-spectral-indices).


```