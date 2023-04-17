# farm_ai/admag/admag_carbon_integration

```yaml

name: admag_carbon_integration
sources:
  baseline_admag_input:
  - baseline_seasonal_field_list.admag_input
  scenario_admag_input:
  - scenario_seasonal_field_list.admag_input
sinks:
  carbon_output: admag_carbon.carbon_output
parameters:
  base_url: null
  client_id: null
  client_secret: null
  authority: null
  default_scope: null
  comet_support_email: null
  ngrok_token: null
tasks:
  baseline_seasonal_field_list:
    op: admag_seasonal_field
    op_dir: admag
    parameters:
      base_url: '@from(base_url)'
      client_id: '@from(client_id)'
      client_secret: '@from(client_secret)'
      authority: '@from(authority)'
      default_scope: '@from(default_scope)'
  scenario_seasonal_field_list:
    op: admag_seasonal_field
    op_dir: admag
    parameters:
      base_url: '@from(base_url)'
      client_id: '@from(client_id)'
      client_secret: '@from(client_secret)'
      authority: '@from(authority)'
      default_scope: '@from(default_scope)'
  admag_carbon:
    op: whatif_comet_local_op
    op_dir: carbon_local
    parameters:
      comet_support_email: '@from(comet_support_email)'
      ngrok_token: '@from(ngrok_token)'
edges:
- origin: baseline_seasonal_field_list.seasonal_field
  destination:
  - admag_carbon.baseline_seasonal_fields
- origin: scenario_seasonal_field_list.seasonal_field
  destination:
  - admag_carbon.scenario_seasonal_fields
description:
  short_description: Computes the offset amount of carbon that would be sequestered
    in a seasonal field using Azure Data Manager for Ag data.
  long_description: Derives carbon sequestration information. Microsoft Azure Data
    Manager for Agriculture (Data Manager for Ag) and the COMET-Farm API are used
    to obtain farming data and evaluate carbon offset.  The Data Manager for Agriculture
    is capable of describing important farming activities such as fertilization, tillage,
    and organic amendments applications, all of which are represented in the data
    manager. FarmVibes.AI retrieves this information from the data manager and builds
    SeasonalFieldInformation FarmVibes.AI objects. These objects are then used to
    call the COMET-Farm API and evaluate Carbon Offset Information.
  sources:
    baseline_admag_input: List of ADMAgSeasonalFieldInput to retrieve SeasonalFieldInformation
      objects for baseline COMET-Farm API Carbon offset evaluation.
    scenario_admag_input: List of ADMAgSeasonalFieldInput to retrieve SeasonalFieldInformation
      objects for scenarios COMET-Farm API Carbon offset evaluation.
  sinks:
    carbon_output: Carbon sequestration received for scenario information provided
      as input.
  parameters:
    comet_support_email: Comet support email. The email used to register for a COMET
      account. The requests are forwarded to comet with this email reference.  This
      email is used by comet to share the information back to you for failed requests.
    ngrok_token: NGROK session token. A token that FarmVibes uses to create a web_hook
      url that is shared with Comet in a request when running the workflow. Comet
      can use this link to send back a response to FarmVibes.  NGROK is a service
      that creates temporary urls for local servers. To use NGROK, FarmVibes needs
      to get a token from this website, https://dashboard.ngrok.com/.
    base_url: Azure Data Manager for Agriculture host. Please visit https://aka.ms/farmvibesDMA
      to check how to get these credentials.
    client_id: Azure Data Manager for Agriculture client id. Please visit https://aka.ms/farmvibesDMA
      to check how to get these credentials.
    client_secret: Azure Data Manager for Agriculture client secret. Please visit
      https://aka.ms/farmvibesDMA to check how to get these credentials.
    authority: Azure Data Manager for Agriculture authority. Please visit https://aka.ms/farmvibesDMA
      to check how to get these credentials.
    default_scope: Azure Data Manager for Agriculture default scope. Please visit
      https://aka.ms/farmvibesDMA to check how to get these credentials.


```

```{mermaid}
    graph TD
    inp1>baseline_admag_input]
    inp2>scenario_admag_input]
    out1>carbon_output]
    tsk1{{baseline_seasonal_field_list}}
    tsk2{{scenario_seasonal_field_list}}
    tsk3{{admag_carbon}}
    tsk1{{baseline_seasonal_field_list}} -- seasonal_field/baseline_seasonal_fields --> tsk3{{admag_carbon}}
    tsk2{{scenario_seasonal_field_list}} -- seasonal_field/scenario_seasonal_fields --> tsk3{{admag_carbon}}
    inp1>baseline_admag_input] -- admag_input --> tsk1{{baseline_seasonal_field_list}}
    inp2>scenario_admag_input] -- admag_input --> tsk2{{scenario_seasonal_field_list}}
    tsk3{{admag_carbon}} -- carbon_output --> out1>carbon_output]
```