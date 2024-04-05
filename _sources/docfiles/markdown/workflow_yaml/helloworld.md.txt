# helloworld

Hello world! Small test workflow that generates an image of the Earth with countries that intersect with the input geometry highlighted in orange.

```{mermaid}
    graph TD
    inp1>user_input]
    out1>raster]
    tsk1{{hello}}
    inp1>user_input] -- user_input --> tsk1{{hello}}
    tsk1{{hello}} -- raster --> out1>raster]
```

## Sources

- **user_input**: Input geometry.

## Sinks

- **raster**: Raster with highlighted countries.

## Tasks

- **hello**: Test op that generates an image of the Earth with countries that intersect with the input geometry highlighted in orange.

## Workflow Yaml

```yaml

name: helloworld
sources:
  user_input:
  - hello.user_input
sinks:
  raster: hello.raster
tasks:
  hello:
    op: helloworld
description:
  short_description: Hello world!
  long_description: Small test workflow that generates an image of the Earth with
    countries that intersect with the input geometry highlighted in orange.
  sources:
    user_input: Input geometry.
  sinks:
    raster: Raster with highlighted countries.


```