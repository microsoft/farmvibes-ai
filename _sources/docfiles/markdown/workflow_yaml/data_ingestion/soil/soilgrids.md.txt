# data_ingestion/soil/soilgrids

```yaml

name: soilgrids
sources:
  input_item:
  - download_soilgrids.input_item
sinks:
  downloaded_raster: download_soilgrids.downloaded_raster
parameters:
  map: wrb
  identifier: MostProbable
tasks:
  download_soilgrids:
    op: download_soilgrids
    parameters:
      map: '@from(map)'
      identifier: '@from(identifier)'
edges: null
description:
  short_description: Downloads digital soil mapping information from SoilGrids for
    the input geometry.
  long_description: The workflow downloads a raster containing the map and identifiers
    for the input geometry. SoilGrids is a system for digital soil mapping based on
    global compilation of soil profile data and environmental layers.
  sources:
    input_item: Input geometry.
  sinks:
    downloaded_raster: Raster with the map and identifiers requested.
  parameters:
    map: "Map to download. Options:\n  - wrb - World Reference Base classes and probabilites\n\
      \  - bdod - Bulk density - kg/dm^3\n  - cec - Cation exchange capacity at ph\
      \ 7 - cmol(c)/kg\n  - cfvo - Coarse fragments volumetric) - cm3/100cm3 (vol%)\n\
      \  - clay - Clay content - g/100g (%)\n  - nitrogen - Nitrogen - g/kg\n  - phh2o\
      \ - Soil pH in H2O - pH\n  - sand - Sand content - g/100g (%)\n  - silt - Silt\
      \ content - g/100g (%)\n  - soc - Soil organic carbon content - g/kg\n  - ocs\
      \ - Soil organic carbon stock - kg/m^3\n  - ocd - Organic carbon densities -\
      \ kg/m^3"
    identifier: "Variable identifier to be downloaded. Depends on map.\n  - wrb: Acrisols,\
      \ Albeluvisols, Alisols, Andosols, Arenosols, Calcisols, Cambisols,\nChernozems,\
      \ Cryosols, Durisols, Ferralsols, Fluvisols, Gleysols, Gypsisols, Histosols,\
      \ Kastanozems, Leptosols, Lixisols, Luvisols, MostProbable, Nitisols, Phaeozems,\
      \ Planosols, Plinthosols, Podzols, Regosols, Solonchaks, Solonetz, Stagnosols,\
      \ Umbrisols, Vertisols.\nOther identifiers follow the nomenclature defined in\
      \ the [link=https://www.isric.org/explore/soilgrids/faq-soilgrids#What_do_the_filename_codes_mean]SoilGrids\
      \ documentation page: https://www.isric.org/explore/soilgrids/faq-soilgrids#What_do_the_filename_codes_mean[/]."


```

```{mermaid}
    graph TD
    inp1>input_item]
    out1>downloaded_raster]
    tsk1{{download_soilgrids}}
    inp1>input_item] -- input_item --> tsk1{{download_soilgrids}}
    tsk1{{download_soilgrids}} -- downloaded_raster --> out1>downloaded_raster]
```