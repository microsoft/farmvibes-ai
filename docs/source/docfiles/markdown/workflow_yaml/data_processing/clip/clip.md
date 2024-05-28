# data_processing/clip/clip

Performs a clip on an input raster based on a provided reference geometry. The workflow outputs a new raster copied from the input raster with its geometry metadata as the intersection between the input raster's geometry and the provided reference geometry. If the parameter hard_clip is set to true, then only data in the intersection is kept in output. The workflow raises an error if there is no intersection between both geometries.

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

## Parameters

- **hard_clip**: if true, keeps only data inside the intersection of reference and input geometries, soft clip otherwise


## Tasks

- **clip_raster**: clips the input raster based on the provided referente geometry.

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
parameters:
  hard_clip: false
tasks:
  clip_raster:
    op: clip_raster
    parameters:
      hard_clip: '@from(hard_clip)'
edges: null
description:
  short_description: Performs a clip on an input raster based on a provided reference
    geometry.
  long_description: The workflow outputs a new raster copied from the input raster
    with its geometry metadata as the intersection between the input raster's geometry
    and the provided reference geometry. If the parameter hard_clip is set to true,
    then only data in the intersection is kept in output. The workflow raises an error
    if there is no intersection between both geometries.
  sources:
    raster: Input raster to be clipped.
    input_geometry: Reference geometry.
  sinks:
    clipped_raster: Clipped raster with the reference geometry.
  parameters:
    hard_clip: 'if true, keeps only data inside the intersection of reference and
      input geometries, soft clip otherwise

      '


```