# data_ingestion/airbus/airbus_price

Prices available AirBus imagery for the input geometry and time range. The workflow will check available imagery, using the AirBus API, that contains the input geometry inside the input time range. The aggregate price (in kB) for matching images will be computed, discounting images already in the user's library. This workflow requires an AirBus API key.

```{mermaid}
    graph TD
    inp1>user_input]
    out1>price]
    tsk1{{list}}
    tsk2{{price}}
    tsk1{{list}} -- airbus_products --> tsk2{{price}}
    inp1>user_input] -- input_item --> tsk1{{list}}
    tsk2{{price}} -- products_price --> out1>price]
```

## Sources

- **user_input**: Time range and geometry of interest.

## Sinks

- **price**: Price for all matching imagery.

## Parameters

- **api_key**: AirBus API key. Required to run the workflow.

## Tasks

- **list**: Lists available AirBus products for the input geometry and time range.

- **price**: Calculates the aggregate price (in kB) for selected AirBus images, discounting images already in the user's library.

## Workflow Yaml

```yaml

name: airbus_price
sources:
  user_input:
  - list.input_item
sinks:
  price: price.products_price
parameters:
  api_key: null
tasks:
  list:
    op: list_airbus_products
    parameters:
      api_key: '@from(api_key)'
  price:
    op: price_airbus_products
    parameters:
      api_key: '@from(api_key)'
edges:
- origin: list.airbus_products
  destination:
  - price.airbus_products
description:
  short_description: Prices available AirBus imagery for the input geometry and time
    range.
  long_description: The workflow will check available imagery, using the AirBus API,
    that contains the input geometry inside the input time range. The aggregate price
    (in kB) for matching images will be computed, discounting images already in the
    user's library. This workflow requires an AirBus API key.
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    price: Price for all matching imagery.
  parameters:
    api_key: AirBus API key. Required to run the workflow.


```