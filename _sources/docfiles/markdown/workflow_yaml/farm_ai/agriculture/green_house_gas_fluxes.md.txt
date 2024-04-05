# farm_ai/agriculture/green_house_gas_fluxes

Computes Green House Fluxes for a region and date range The workflow follows the GHG Protocol guidelines published for Brazil (which are based on IPCC reports) to compute Green House Gas emission fluxes (sequestration versus emissions) for a given crop.

```{mermaid}
    graph TD
    inp1>user_input]
    out1>fluxes]
    tsk1{{ghg}}
    inp1>user_input] -- ghg --> tsk1{{ghg}}
    tsk1{{ghg}} -- fluxes --> out1>fluxes]
```

## Sources

- **user_input**: The user-provided inputs for GHG computation.

## Sinks

- **fluxes**: The computed fluxes for the given area and date range considering the user input data.

## Parameters

- **crop_type**: The type of the crop to compute GHG emissions. Supported crops are 'wheat', 'corn', 'cotton', and 'soybeans'.

## Tasks

- **ghg**: Computes Green House Gas emission fluxes based on emission factors based on IPCC methodology.

## Workflow Yaml

```yaml

name: green_house_gas_fluxes
sources:
  user_input:
  - ghg.ghg
sinks:
  fluxes: ghg.fluxes
parameters:
  crop_type: corn
tasks:
  ghg:
    op: compute_ghg_fluxes
    parameters:
      crop_type: '@from(crop_type)'
edges: null
description:
  short_description: Computes Green House Fluxes for a region and date range
  long_description: The workflow follows the GHG Protocol guidelines published for
    Brazil (which are based on IPCC reports) to compute Green House Gas emission fluxes
    (sequestration versus emissions) for a given crop.
  sources:
    user_input: The user-provided inputs for GHG computation.
  sinks:
    fluxes: The computed fluxes for the given area and date range considering the
      user input data.
  parameters:
    crop_type: The type of the crop to compute GHG emissions. Supported crops are
      'wheat', 'corn', 'cotton', and 'soybeans'.


```