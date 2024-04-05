# data_ingestion/admag/prescriptions

Fetches prescriptions using ADMAg (Microsoft Azure Data Manager for Agriculture). The workflow fetch prescriptions (sensor samples) linked to prescription_map_id. Each sensor sample have the information of nutrient (Nitrogen, Carbon, Phosphorus, pH, Latitude, Longitude etc., ). The Latitude & Longitude used to create a point geometry. Geometry and nutrient information transformed to GeoJSON. The GeoJSON stored as asset in farmvibes-ai.

```{mermaid}
    graph TD
    inp1>admag_input]
    out1>response]
    tsk1{{list_prescriptions}}
    tsk2{{get_prescription}}
    tsk3{{admag_prescriptions}}
    tsk1{{list_prescriptions}} -- prescriptions/prescription_without_geom_input --> tsk2{{get_prescription}}
    tsk2{{get_prescription}} -- prescription_with_geom/prescriptions_with_geom_input --> tsk3{{admag_prescriptions}}
    inp1>admag_input] -- admag_input --> tsk1{{list_prescriptions}}
    inp1>admag_input] -- admag_input --> tsk3{{admag_prescriptions}}
    tsk3{{admag_prescriptions}} -- response --> out1>response]
```

## Sources

- **admag_input**: Required inputs to access ADMAg resources, party_id and prescription_map_id that helps fetching prescriptions.

## Sinks

- **response**: Prescriptions received from ADMAg.

## Parameters

- **base_url**: URL to access the registered app. Refer this url to create required resources for admag. https://learn.microsoft.com/en-us/azure/data-manager-for-agri/quickstart-install-data-manager-for-agriculture

- **client_id**: Value uniquely identifies registered application in the Microsoft identity platform. Visit url https://learn.microsoft.com/en-us/azure/data-manager-for-agri/quickstart-install-data-manager-for-agriculture to register the app.

- **client_secret**: Sometimes called an application password, a client secret is a string value your app can use in place of a certificate to identity itself.

- **authority**: The endpoint URIs for your app are generated automatically when you register or configure your app. It is used by client to obtain authorization from the resource owner

- **default_scope**: URL for default azure OAuth2 permissions

## Tasks

- **list_prescriptions**: List available prescriptions using prescription map.

- **get_prescription**: Get prescription using ADMAg API.

- **admag_prescriptions**: Downloads boundary and prescriptions linked to seasonal field from ADMAg data source.

## Workflow Yaml

```yaml

name: admag_prescritpions
sources:
  admag_input:
  - list_prescriptions.admag_input
  - admag_prescriptions.admag_input
sinks:
  response: admag_prescriptions.response
parameters:
  base_url: null
  client_id: null
  client_secret: null
  authority: null
  default_scope: null
tasks:
  list_prescriptions:
    op: list_prescriptions
    op_dir: admag
    parameters:
      base_url: '@from(base_url)'
      client_id: '@from(client_id)'
      client_secret: '@from(client_secret)'
      authority: '@from(authority)'
      default_scope: '@from(default_scope)'
  get_prescription:
    op: get_prescription
    op_dir: admag
    parameters:
      base_url: '@from(base_url)'
      client_id: '@from(client_id)'
      client_secret: '@from(client_secret)'
      authority: '@from(authority)'
      default_scope: '@from(default_scope)'
  admag_prescriptions:
    op: prescriptions
    op_dir: admag
    parameters:
      base_url: '@from(base_url)'
      client_id: '@from(client_id)'
      client_secret: '@from(client_secret)'
      authority: '@from(authority)'
      default_scope: '@from(default_scope)'
edges:
- origin: list_prescriptions.prescriptions
  destination:
  - get_prescription.prescription_without_geom_input
- origin: get_prescription.prescription_with_geom
  destination:
  - admag_prescriptions.prescriptions_with_geom_input
description:
  short_description: Fetches prescriptions using ADMAg (Microsoft Azure Data Manager
    for Agriculture).
  long_description: The workflow fetch prescriptions (sensor samples) linked to prescription_map_id.
    Each sensor sample have the information of nutrient (Nitrogen, Carbon, Phosphorus,
    pH, Latitude, Longitude etc., ). The Latitude & Longitude used to create a point
    geometry. Geometry and nutrient information transformed to GeoJSON. The GeoJSON
    stored as asset in farmvibes-ai.
  sources:
    admag_input: Required inputs to access ADMAg resources, party_id and prescription_map_id
      that helps fetching prescriptions.
  sinks:
    response: Prescriptions received from ADMAg.
  parameters:
    base_url: URL to access the registered app. Refer this url to create required
      resources for admag. https://learn.microsoft.com/en-us/azure/data-manager-for-agri/quickstart-install-data-manager-for-agriculture
    client_id: Value uniquely identifies registered application in the Microsoft identity
      platform. Visit url https://learn.microsoft.com/en-us/azure/data-manager-for-agri/quickstart-install-data-manager-for-agriculture
      to register the app.
    client_secret: Sometimes called an application password, a client secret is a
      string value your app can use in place of a certificate to identity itself.
    authority: The endpoint URIs for your app are generated automatically when you
      register or configure your app. It is used by client to obtain authorization
      from the resource owner
    default_scope: URL for default azure OAuth2 permissions


```