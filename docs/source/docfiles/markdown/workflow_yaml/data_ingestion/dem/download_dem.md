# data_ingestion/dem/download_dem

Downloads digital elevation map tiles that intersect with the input geometry and time range. The workflow will download digital elevation maps from the USGS 3DEP datasets (available for the United States at 10 and 30 meters) or Copernicus DEM GLO-30 (globally at 30 meters) through the Planetary Computer. For more information, see https://planetarycomputer.microsoft.com/dataset/3dep-seamless and https://planetarycomputer.microsoft.com/dataset/cop-dem-glo-30 .

```{mermaid}
    graph TD
    inp1>user_input]
    out1>raster]
    tsk1{{list}}
    tsk2{{download}}
    tsk1{{list}} -- dem_products/input_product --> tsk2{{download}}
    inp1>user_input] -- input_items --> tsk1{{list}}
    tsk2{{download}} -- downloaded_product --> out1>raster]
```

## Sources

- **user_input**: Time range and geometry of interest.

## Sinks

- **raster**: DEM raster.

## Parameters

- **pc_key**: Optional Planetary Computer API key.

- **resolution**: Spatial resolution of the DEM. 10m and 30m are available.

- **provider**: Provider of the DEM. "USGS3DEP" and "CopernicusDEM30" are available.

## Tasks

- **list**: Lists digital elevation map tiles that intersect with the input geometry and time range.

- **download**: Downloads digital elevation map raster given a DemProduct.

## Workflow Yaml

```yaml

name: download_dem
sources:
  user_input:
  - list.input_items
sinks:
  raster: download.downloaded_product
parameters:
  pc_key: null
  resolution: 10
  provider: USGS3DEP
tasks:
  list:
    op: list_dem_products
    parameters:
      resolution: '@from(resolution)'
      provider: '@from(provider)'
  download:
    op: download_dem
    parameters:
      api_key: '@from(pc_key)'
edges:
- origin: list.dem_products
  destination:
  - download.input_product
description:
  short_description: Downloads digital elevation map tiles that intersect with the
    input geometry and time range.
  long_description: The workflow will download digital elevation maps from the USGS
    3DEP datasets (available for the United States at 10 and 30 meters) or Copernicus
    DEM GLO-30 (globally at 30 meters) through the Planetary Computer. For more information,
    see https://planetarycomputer.microsoft.com/dataset/3dep-seamless and https://planetarycomputer.microsoft.com/dataset/cop-dem-glo-30
    .
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    raster: DEM raster.
  parameters:
    pc_key: Optional Planetary Computer API key.
    resolution: Spatial resolution of the DEM. 10m and 30m are available.
    provider: Provider of the DEM. "USGS3DEP" and "CopernicusDEM30" are available.


```