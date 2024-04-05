# data_ingestion/hansen/hansen_forest_change_download

Downloads and merges Global Forest Change (Hansen) rasters that intersect the user-provided geometry/time range. The workflow lists Global Forest Change (Hansen) products that intersect the user-provided geometry/time range, downloads the data for each of them, and merges the rasters. The dataset is available at 30m resolution and is updated annually. The data contains information on forest cover, loss, and gain. The default dataset version is GFC-2022-v1.10 and is passed to the workflow as the parameter tiles_folder_url. For the default version, the dataset is available from 2000 to 2022.  Dataset details can be found at https://storage.googleapis.com/earthenginepartners-hansen/GFC-2022-v1.10/download.html.

```{mermaid}
    graph TD
    inp1>input_item]
    out1>merged_raster]
    out2>downloaded_raster]
    tsk1{{list}}
    tsk2{{download}}
    tsk3{{group}}
    tsk4{{merge}}
    tsk1{{list}} -- hansen_products/hansen_product --> tsk2{{download}}
    tsk2{{download}} -- raster/rasters --> tsk3{{group}}
    tsk3{{group}} -- raster_groups/raster_sequence --> tsk4{{merge}}
    inp1>input_item] -- input_item --> tsk1{{list}}
    tsk4{{merge}} -- raster --> out1>merged_raster]
    tsk2{{download}} -- raster --> out2>downloaded_raster]
```

## Sources

- **input_item**: User-provided geometry and time range.

## Sinks

- **merged_raster**: Merged Global Forest Change (Hansen) data as a raster.

- **downloaded_raster**: Individual Global Forest Change (Hansen) rasters prior to the merge operation.

## Parameters

- **layer_name**: Name of the Global Forest Change (Hansen) layer. Can be any of the following names 'treecover2000', 'loss', 'gain', 'lossyear', 'datamask', 'first', 'last'.

- **tiles_folder_url**: URL to the Global Forest Change (Hansen) dataset. It specifies the dataset version and is used to download the data.

## Tasks

- **list**: Lists Global Forest Change (Hansen) products that intersect the user-provided geometry/time range.

- **download**: Downloads Global Forest Change (Hansen) data.

- **group**: This op groups rasters in time according to 'criterion'.

- **merge**: Merges rasters in a sequence to a single raster.

## Workflow Yaml

```yaml

name: glad_forest_change_download
sources:
  input_item:
  - list.input_item
sinks:
  merged_raster: merge.raster
  downloaded_raster: download.raster
parameters:
  layer_name: null
  tiles_folder_url: https://storage.googleapis.com/earthenginepartners-hansen/GFC-2022-v1.10/
tasks:
  list:
    op: list_hansen_products
    parameters:
      tiles_folder_url: '@from(tiles_folder_url)'
      layer_name: '@from(layer_name)'
  download:
    op: download_hansen
  group:
    op: group_rasters_by_time
    parameters:
      criterion: year
  merge:
    op: merge_rasters
edges:
- origin: list.hansen_products
  destination:
  - download.hansen_product
- origin: download.raster
  destination:
  - group.rasters
- origin: group.raster_groups
  destination:
  - merge.raster_sequence
description:
  short_description: Downloads and merges Global Forest Change (Hansen) rasters that
    intersect the user-provided geometry/time range.
  long_description: The workflow lists Global Forest Change (Hansen) products that
    intersect the user-provided geometry/time range, downloads the data for each of
    them, and merges the rasters. The dataset is available at 30m resolution and is
    updated annually. The data contains information on forest cover, loss, and gain.
    The default dataset version is GFC-2022-v1.10 and is passed to the workflow as
    the parameter tiles_folder_url. For the default version, the dataset is available
    from 2000 to 2022.  Dataset details can be found at https://storage.googleapis.com/earthenginepartners-hansen/GFC-2022-v1.10/download.html.
  sources:
    input_item: User-provided geometry and time range.
  sinks:
    merged_raster: Merged Global Forest Change (Hansen) data as a raster.
    downloaded_raster: Individual Global Forest Change (Hansen) rasters prior to the
      merge operation.
  parameters:
    tiles_folder_url: URL to the Global Forest Change (Hansen) dataset. It specifies
      the dataset version and is used to download the data.
    layer_name: Name of the Global Forest Change (Hansen) layer. Can be any of the
      following names 'treecover2000', 'loss', 'gain', 'lossyear', 'datamask', 'first',
      'last'.


```