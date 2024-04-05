# REST API

Once the FarmVibes.AI cluster is up and running, you can interact with it using the REST API, which provides a set of endpoints that allow you to list and describe workflows, as well as manage workflow runs.
The REST API is available at the URL and port specified during cluster creation, and its address is printed in the terminal once the setup is complete. You can also check the address by running the following command in the terminal:

```bash
$ farmvibes-ai <local | remote> status
2024-01-01 00:00:00,000 - INFO    - Cluster farmvibes-ai-username is running with 1 servers and 0 agents.
2024-01-01 00:00:00,001 - INFO    - Service url is http://ip.address:port
```

## Interacting with the API

The API is accessible from the [FarmVibes.AI Python client](https://microsoft.github.io/farmvibes-ai/docfiles/markdown/CLIENT.html), which provides an interface to interact with the cluster, list workflows, and manage workflow runs.
Alternativelly, interacting with the API can be done using any tool that can send HTTP requests, such as `curl` or [Bruno](https://www.usebruno.com/).

For example, to list the available workflows, you can use the following command:

```bash
$ curl -X GET http://localhost:31108/v0/workflows
```

Which will return the following list:

```
["helloworld","farm_ai/land_degradation/landsat_ndvi_trend","farm_ai/land_degradation/ndvi_linear_trend", ...]
```

For submiting a run of a specific workflow, we need to pass a JSON with the run configuration
(i.e., workflow name, input geometry and time range, workflow parameters, etc) as the body of the
request. For example, we can use the following command to create a `helloworld` workflow run:

```bash
$ curl -X POST -H "Content-Type: application/json" -d <JSON>
```

Replacing the body of the request `<JSON>` with the following:

```json
{
  "name": "Hello!",
  "workflow": "helloworld",
  "parameters": null,
  "user_input": {
    "start_date": "2020-05-01T00:00:00",
    "end_date": "2020-05-05T00:00:00",
    "geojson": {
      "features": [
        {
          "geometry": {
            "type": "Polygon",
            "coordinates": [
              [
                [
                  -119.14896203939314,
                  46.51578909859286
                ],
                [
                  -119.14896203939314,
                  46.37578909859286
                ],
                [
                  -119.28896203939313,
                  46.37578909859286
                ],
                [
                  -119.28896203939313,
                  46.51578909859286
                ],
                [
                  -119.14896203939314,
                  46.51578909859286
                ]
              ]
            ]
          },
          "type": "Feature"
        }
      ],
      "type": "FeatureCollection"
    }
  }
}
```

To help in understanding the expected format and structure of the json in our requests, we provide in
our Python client the `_form_payload` method ([`vibe_core.client.FarmvibesAiClient._form_payload`](https://microsoft.github.io/farmvibes-ai/docfiles/code/vibe_core_client/client.html#vibe_core.client.FarmvibesAiClient._form_payload)) that can be used to
generate the request payload for a given run configuration. For example, the following code could
be used to obtain the json above for the helloworld workflow:

```python
from vibe_core.client import get_default_vibe_client
import shapely.geometry as shpg
from datetime import datetime

client = get_default_vibe_client()

geom = shpg.Point(-119.21896203939313, 46.44578909859286).buffer(.07, cap_style=3)
time_range = (datetime(2020, 5, 1), datetime(2020, 5, 5))

payload = client._form_payload("helloworld", None, geom, time_range, None,"Hello!")
```

Another example, considering the `farm_ai/segmentation/segment_s2` workflow run submited in the
[Sentinel-2 Segmentation notebook](https://github.com/microsoft/farmvibes-ai/blob/main/notebooks/segment_anything/sentinel2_segmentation.ipynb), would be:

```python
payload = client._form_payload("farm_ai/segmentation/segment_s2", None, None, None, {"user_input": roi_time_range, "prompts": geom_collection},"SAM segmentation") 
```

Which would generate the following json:

```json
{
  "name": "SAM segmentation",
  "workflow": "farm_ai/segmentation/segment_s2",
  "parameters": null,
  "user_input": {
    "user_input": {
      "type": "Feature",
      "stac_version": "1.0.0",
      "id": "f6465ad0-5e01-4792-ad99-a0bd240c1e7d",
      "properties": {
        "start_datetime": "2020-05-01T00:00:00+00:00",
        "end_datetime": "2020-05-05T00:00:00+00:00",
        "datetime": "2020-05-01T00:00:00Z"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              -119.14896203939314,
              46.51578909859286
            ],
            [
              -119.14896203939314,
              46.37578909859286
            ],
            [
              -119.28896203939313,
              46.37578909859286
            ],
            [
              -119.28896203939313,
              46.51578909859286
            ],
            [
              -119.14896203939314,
              46.51578909859286
            ]
          ]
        ]
      },
      "links": [],
      "assets": {},
      "bbox": [
        -119.28896203939313,
        46.37578909859286,
        -119.14896203939314,
        46.51578909859286
      ],
      "stac_extensions": [],
      "terravibes_data_type": "DataVibe"
    },
    "prompts": {
      "type": "Feature",
      "stac_version": "1.0.0",
      "id": "geo_734c6441-cb25-4c40-8204-6b7286f24bb9",
      "properties": {
        "urls": [
          "/mnt/734c6441-cb25-4c40-8204-6b7286f24bb9_geometry_collection.geojson"
        ],
        "start_datetime": "2020-05-01T00:00:00+00:00",
        "end_datetime": "2020-05-05T00:00:00+00:00",
        "datetime": "2020-05-01T00:00:00Z"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              -119.14896203939314,
              46.51578909859286
            ],
            [
              -119.14896203939314,
              46.37578909859286
            ],
            [
              -119.28896203939313,
              46.37578909859286
            ],
            [
              -119.28896203939313,
              46.51578909859286
            ],
            [
              -119.14896203939314,
              46.51578909859286
            ]
          ]
        ]
      },
      "links": [],
      "assets": {},
      "bbox": [
        -119.28896203939313,
        46.37578909859286,
        -119.14896203939314,
        46.51578909859286
      ],
      "stac_extensions": [],
      "terravibes_data_type": "ExternalReferenceList"
    }
  }
}
```

For more information about the `_form_payload` method, please refer to the [FarmVibes.AI Python client documentation](https://microsoft.github.io/farmvibes-ai/docfiles/code/vibe_core_client/client.html#vibe_core.client.FarmvibesAiClient._form_payload).

## Endpoints

We provide below a list of the available endpoints and their descriptions.

-----------------------------

```{eval-rst}
.. openapi:: ../openapi.json
    :examples:
    :format: markdown
```
