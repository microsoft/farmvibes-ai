# data_processing/clip/clip

Performs a soft clip on an input raster based on a provided reference geometry. The workflow outputs a new raster copied from the input raster with its geometry metadata as the intersection between the input raster's geometry and the provided reference geometry. The workflow raises an error if there is no intersection between both geometries.

```{mermaid}
    graph TD
    inp1>raster]
    inp2>input_geometry]
    out1>clipped_raster]
    tsk1{{clip_raster}}
    inp1>raster] -- raster --> tsk1{{clip_raster}}
    inp2>input_geometry] -- input_item --> tsk1{{clip_raster}}
    tsk1{{clip_raster}} -- clipped_raster --> out1>clipped_raster]
```

## Sources

- **raster**: Input raster to be clipped.

- **input_geometry**: Reference geometry.

## Sinks

- **clipped_raster**: Clipped raster with the reference geometry.

## Tasks

- **clip_raster**: Soft clips the input raster based on the provided referente geometry.

## Workflow Yaml

```yaml

name: clip
sources:
  raster:
  - clip_raster.raster
  input_geometry:
  - clip_raster.input_item
sinks:
  clipped_raster: clip_raster.clipped_raster
tasks:
  clip_raster:
    op: clip_raster
edges: null
description:
  short_description: Performs a soft clip on an input raster based on a provided reference
    geometry.
  long_description: The workflow outputs a new raster copied from the input raster
    with its geometry metadata as the intersection between the input raster's geometry
    and the provided reference geometry. The workflow raises an error if there is
    no intersection between both geometries.
  sources:
    raster: Input raster to be clipped.
    input_geometry: Reference geometry.
  sinks:
    clipped_raster: Clipped raster with the reference geometry.
  parameters: null


```