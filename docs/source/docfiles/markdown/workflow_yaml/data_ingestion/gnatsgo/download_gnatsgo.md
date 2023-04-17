# data_ingestion/gnatsgo/download_gnatsgo

```yaml

name: download_gnatsgo
sources:
  user_input:
  - list.input_item
sinks:
  raster: download.downloaded_raster
parameters:
  pc_key: null
  variable: soc0_5
tasks:
  list:
    op: list_gnatsgo_products
  download:
    op: download_gnatsgo
    parameters:
      api_key: '@from(pc_key)'
      variable: '@from(variable)'
edges:
- origin: list.gnatsgo_products
  destination:
  - download.gnatsgo_product
description:
  short_description: Downloads gNATSGO raster data that intersect with the input geometry
    and time range.
  long_description: This workflow lists and downloads raster products of gNATSGO dataset
    from Planetary Computer. Input geometry must fall within Continel USA, whereas
    input time range can be arbitrary (all gNATSGO assets are from 2020-07-01). For
    more information on the available properties, see https://planetarycomputer.microsoft.com/dataset/gnatsgo-rasters.
  sources:
    user_input: Geometry of interest (arbitrary time range).
  sinks:
    raster: Raster with desired property.
  parameters:
    pc_key: Optional Planetary Computer API key.
    variable: "Options are:\n  aws{DEPTH} - Available water storage estimate (AWS)\
      \ for the DEPTH zone.\n  soc{DEPTH} - Soil organic carbon stock estimate (SOC)\
      \ for the DEPTH zone.\n  tk{DEPTH}a - Thickness of soil components used in the\
      \ DEPTH zone for the AWS calculation.\n  tk{DEPTH}s - Thickness of soil components\
      \ used in the DEPTH zone for the SOC calculation.\n  mukey - Map unit key, a\
      \ unique identifier of a record for matching with gNATSGO tables.\n  droughty\
      \ - Drought vulnerability estimate.\n  nccpi3all - National Commodity Crop Productivity\
      \ Index that has the highest value among Corn\nand Soybeans, Small Grains, or\
      \ Cotton for major earthy components.\n  nccpi3corn - National Commodity Crop\
      \ Productivity Index for Corn for major earthy\ncomponents.\n  nccpi3cot - National\
      \ Commodity Crop Productivity Index for Cotton for major earthy\ncomponents.\n\
      \  nccpi3sg - National Commodity Crop Productivity Index for Small Grains for\
      \ major earthy\ncomponents.\n  nccpi3soy - National Commodity Crop Productivity\
      \ Index for Soy for major earthy components.\n  pctearthmc - National Commodity\
      \ Crop Productivity Index map unit percent earthy is the map\nunit summed comppct_r\
      \ for major earthy components.\n  pwsl1pomu - Potential Wetland Soil Landscapes\
      \ (PWSL).\n  rootznaws - Root zone (commodity crop) available water storage\
      \ estimate (RZAWS).\n  rootznemc - Root zone depth is the depth within the soil\
      \ profile that commodity crop (cc)\nroots can effectively extract water and\
      \ nutrients for growth.\n  musumcpct - Sum of the comppct_r (SSURGO component\
      \ table) values for all listed components\nin the map unit.\n  musumcpcta -\
      \ Sum of the comppct_r (SSURGO component table) values used in the available\n\
      water storage calculation for the map unit.\n  musumcpcts - Sum of the comppct_r\
      \ (SSURGO component table) values used in the soil organic\ncarbon calculation\
      \ for the map unit. \ngNATSGO has properties available for multiple soil depths.\
      \ You may exchange DEPTH in the variable names above for any of the following\
      \ (all measured in cm): \n  0_5\n  0_20\n  0_30\n  5_20\n  0_100\n  0_150\n\
      \  0_999\n  20_50\n  50_100\n  100_150\n  150_999"


```

```{mermaid}
    graph TD
    inp1>user_input]
    out1>raster]
    tsk1{{list}}
    tsk2{{download}}
    tsk1{{list}} -- gnatsgo_products/gnatsgo_product --> tsk2{{download}}
    inp1>user_input] -- input_item --> tsk1{{list}}
    tsk2{{download}} -- downloaded_raster --> out1>raster]
```