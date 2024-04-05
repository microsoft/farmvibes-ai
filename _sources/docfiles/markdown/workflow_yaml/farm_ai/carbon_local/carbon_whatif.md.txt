# farm_ai/carbon_local/carbon_whatif

Computes the offset amount of carbon that would be sequestered in a seasonal field using the baseline (historical) and scenario (time range interested in) information. To derive amount of carbon, it relies on seasonal information information provided for both baseline and scenario. The baseline represents historical information of farm practices used during each season that includes fertilizers, tillage, harvest and organic amendment. Minimum 2 years of baseline information required to execute the workflow. The scenario represents future farm practices planning to do during each season that includes fertilizers, tillage, harvest and organic amendment. For the scenario information provided, the workflow compute the offset amount of carbon that would be sequestrated in a seasonal field. Minimum 2years of baseline information required to execute the workflow. The requests received by workflow are forwarded to comet api. To know more information of comet refer to https://gitlab.com/comet-api/api-docs/-/tree/master/. To understand the enumerations and information accepted by comet refer to https://gitlab.com/comet-api/api-docs/-/blob/master/COMET-Farm_API_File_Specification.xlsx The request submitted get executed with in 5 minutes to max 2 hours. If response not received from comet within this time period, check comet_support_email for information on failed requests, if no emails received check status of requests by contacting to this support email address of comet "appnrel@colostate.edu". For public use comet limits 50 requests each day. If more requests need to send contact support email address.

```{mermaid}
    graph TD
    inp1>baseline_seasonal_fields]
    inp2>scenario_seasonal_fields]
    out1>carbon_output]
    tsk1{{comet_task}}
    inp1>baseline_seasonal_fields] -- baseline_seasonal_fields --> tsk1{{comet_task}}
    inp2>scenario_seasonal_fields] -- scenario_seasonal_fields --> tsk1{{comet_task}}
    tsk1{{comet_task}} -- carbon_output --> out1>carbon_output]
```

## Sources

- **baseline_seasonal_fields**: List of seasonal fields that holds the historical information of farm practices such as fertilizers, tillage, harvest and organic amendment.

- **scenario_seasonal_fields**: List of seasonal fields that holds the future information of farm practices such as fertilizers, tillage, harvest and organic amendment.

## Sinks

- **carbon_output**: Carbon sequestration received for scenario information provided as input.

## Parameters

- **comet_support_email**: COMET-Farm API Registered email. The requests are forwarded to comet with this email reference. This email used by comet to share the information back to you for failed requests.

- **ngrok_token**: NGROK session token. FarmVibes generate web_hook url and shared url with comet along the request to receive the response from comet. It's publicly accessible url and it's unique for each session. The url gets destroyed once the session ends. To start the ngrok session a token, it is generated from this url https://dashboard.ngrok.com/

## Tasks

- **comet_task**: Computes the offset amount of carbon that would be sequestered in a seasonal field using the baseline (historical) and scenario (time range interested in) information.

## Workflow Yaml

```yaml

name: carbon_whatif
sources:
  baseline_seasonal_fields:
  - comet_task.baseline_seasonal_fields
  scenario_seasonal_fields:
  - comet_task.scenario_seasonal_fields
sinks:
  carbon_output: comet_task.carbon_output
parameters:
  comet_support_email: null
  ngrok_token: null
tasks:
  comet_task:
    op: whatif_comet_local_op
    op_dir: carbon_local
    parameters:
      comet_support_email: '@from(comet_support_email)'
      ngrok_token: '@from(ngrok_token)'
description:
  short_description: Computes the offset amount of carbon that would be sequestered
    in a seasonal field using the baseline (historical) and scenario (time range interested
    in) information.
  long_description: To derive amount of carbon, it relies on seasonal information
    information provided for both baseline and scenario. The baseline represents historical
    information of farm practices used during each season that includes fertilizers,
    tillage, harvest and organic amendment. Minimum 2 years of baseline information
    required to execute the workflow. The scenario represents future farm practices
    planning to do during each season that includes fertilizers, tillage, harvest
    and organic amendment. For the scenario information provided, the workflow compute
    the offset amount of carbon that would be sequestrated in a seasonal field. Minimum
    2years of baseline information required to execute the workflow. The requests
    received by workflow are forwarded to comet api. To know more information of comet
    refer to https://gitlab.com/comet-api/api-docs/-/tree/master/. To understand the
    enumerations and information accepted by comet refer to https://gitlab.com/comet-api/api-docs/-/blob/master/COMET-Farm_API_File_Specification.xlsx
    The request submitted get executed with in 5 minutes to max 2 hours. If response
    not received from comet within this time period, check comet_support_email for
    information on failed requests, if no emails received check status of requests
    by contacting to this support email address of comet "appnrel@colostate.edu".
    For public use comet limits 50 requests each day. If more requests need to send
    contact support email address.
  sources:
    baseline_seasonal_fields: List of seasonal fields that holds the historical information
      of farm practices such as fertilizers, tillage, harvest and organic amendment.
    scenario_seasonal_fields: List of seasonal fields that holds the future information
      of farm practices such as fertilizers, tillage, harvest and organic amendment.
  sinks:
    carbon_output: Carbon sequestration received for scenario information provided
      as input.
  parameters:
    comet_support_email: COMET-Farm API Registered email. The requests are forwarded
      to comet with this email reference. This email used by comet to share the information
      back to you for failed requests.
    ngrok_token: NGROK session token. FarmVibes generate web_hook url and shared url
      with comet along the request to receive the response from comet. It's publicly
      accessible url and it's unique for each session. The url gets destroyed once
      the session ends. To start the ngrok session a token, it is generated from this
      url https://dashboard.ngrok.com/


```