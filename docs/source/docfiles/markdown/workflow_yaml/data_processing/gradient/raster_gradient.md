# data_processing/gradient/raster_gradient

```yaml

name: raster_gradient
sources:
  raster:
  - gradient.input_raster
sinks:
  gradient: gradient.output_raster
tasks:
  gradient:
    op: compute_raster_gradient
edges: null
description:
  short_description: Computes the gradient of each band of the input raster with a
    Sobel operator.
  long_description: null
  sources:
    raster: Input raster.
  sinks:
    gradient: Raster with the gradients.
  parameters: null


```

```{mermaid}
    graph TD
    inp1>raster]
    out1>gradient]
    tsk1{{gradient}}
    inp1>raster] -- input_raster --> tsk1{{gradient}}
    tsk1{{gradient}} -- output_raster --> out1>gradient]
```