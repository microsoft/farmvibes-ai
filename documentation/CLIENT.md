# FarmVibes.AI Python Client User Manual

We provide a python client for interacting with the cluster.
The client is accessible by installing the `vibe_core` package, and can be instantiated with:

```python
from vibe_core.client import get_default_vibe_client
client = get_default_vibe_client()
```

The `get_default_vibe_client` function will automatically target your local minikube cluster.

## Checking available workflows

The `list_workflows` method can be used to list the name of all available workflows

```python
>>> client.list_workflows()[:7]
['ingest_raster',
'helloworld',
'farm_ai/land_cover_mapping/conservation_practices',
'farm_ai/agriculture/canopy_cover',
'farm_ai/agriculture/methane_index',
'farm_ai/agriculture/emergence_summary',
'farm_ai/agriculture/change_detection',
]
```

The `document_workflow` method provides documentation about a workflow, including it's inputs, outputs,
and parameters.

```text
>>> client.document_workflow("helloworld")

Workflow: helloworld

Description:
    Hello world! Small test workflow that generates an image of the Earth with countries that
    intersect with the input geometry highlighted in orange.

Sources:
    - user_input (vibe_core.data.core_types.DataVibe): Input geometry.

Sinks:
    - raster (vibe_core.data.rasters.Raster): Raster with highlighted countries.

Tasks:
    - hello
```

For more information about workflows, check [the workflow documentation](./WORKFLOWS.md).

## Submitting a workflow run

To submit a workflow run, use the `run` method, which takes in the workflow, a run name,
the workflow input, and optional parameter overrides. For workflows that take in a `DataVibe`
defining a region and time range of interest, the inputs can be a geometry (`shapely` object)
and time range (tuple of `datetime` objects). See the example below:

```python
from shapely import geometry as shpg
from datetime import datetime
# Geometry in WGS-84/EPSG:4326
geom = shpg.box(-122.142363,47.681775, -122.106146, 47.667801)
# Time range with start and end
time_range = (datetime(2020, 1, 1), datetime(2022, 1, 1))
run = client.run("helloworld", "My first workflow run", geometry=geom, time_range=time_range)
```

To submit a run with other inputs, use the alternate `run` signature. The following is an equivalent
example of the previous run, but instead submitting a `DataVibe` object.

```python
from vibe_core.data import DataVibe, gen_guid
vibe_input = DataVibe(id=gen_guid(), geometry=shpg.mapping(geom), time_range=time_range, assets=[])
# Since this workflow only has a single source (input), we can pass in the object directly, and
# it will be assigned to the only source
run = client.run("helloworld", "Workflow run with other inputs", input_data=vibe_input)
# More generally, pass in a dict where the keys are the workflow sources
run = client.run(
    "helloworld",
    "Workflow run with other inputs",
    input_data={"user_input": vibe_input}
)
```

## Checking on your workflow run

The `run` method will return a `VibeWorkflowRun` object, that contains information about your run,
and can be used to keep track of your run progress, access outputs, and more. The object
representation will display the run id, name, workflow, and status:

```python
>>> run
VibeWorkflowRun(id='7b95932f-2428-4036-b4cc-14ef832bf8c2', name='My first workflow run', workflow='helloworld', status='running')
```

This information can also be queried with their respective property. For the status, it will be
refreshed at every call:

```python
>>> run.status
<RunStatus.done: 'done'>
```

For more detailed information about each task in the workflow run, use `task_status`, and
`task_details`:

```python
>>> run.task_status  # Status of each task
{'hello': 'done', 'output_handler': 'done'}
>>> run.task_details  # Full details
{'hello': RunDetails(start_time=datetime.datetime(2022, 10, 3, 22, 22, 4, 609784), end_time=datetime.datetime(2022, 10, 3, 22, 22, 9, 533641), reason=None, status='done'),
 'output_handler': RunDetails(start_time=datetime.datetime(2022, 10, 3, 22, 22, 9, 556604), end_time=datetime.datetime(2022, 10, 3, 22, 22, 9, 594992), reason=None, status='done')}
```

To monitor the run in a continuous manner, use the `monitor` method. It will draw a table on the
terminal and update it on a regular interval

```text
>>> run.monitor()
                         🌎 FarmVibes.AI 🌍 helloworld 🌏
                   Run id: 7b95932f-2428-4036-b4cc-14ef832bf8c2
┏━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Task Name      ┃ Status ┃ Start Time          ┃ End Time            ┃ Duration ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ output_handler │ done   │ 2022/10/03 22:22:09 │ 2022/10/03 22:22:09 │ 00:00:00 │
│ hello          │ done   │ 2022/10/03 22:22:04 │ 2022/10/03 22:22:09 │ 00:00:04 │
└────────────────┴────────┴─────────────────────┴─────────────────────┴──────────┘
                         Last update: 2022/10/03 22:23:59
```

The run call is asynchronous: the cluster will start working on your submission, but the interpreter
is free. To block the interpreter (*e.g.*, in a script that needs to wait for a run to be done),
use `block_until_complete()`. You can optionally define a timeout in seconds:

```python
run.block_until_complete()  # Will wait until the run is done
run.block_until_complete(timeout=60)  # Will raise RuntimeError if the run is not done in 60s
```

### Checking previous runs

To list all runs, use the `list_runs` method. It will return a list of `ids` for all runs in the
cluster. Run ids follow the [UUID4 standard](https://datatracker.ietf.org/doc/html/rfc4122.html).

```python
>>> client.list_runs()
['7b95932f-2428-4036-b4cc-14ef832bf8c2']
```

You can then obtain a `VibeWorkflowRun` from the id with the `get_run_by_id` method

```python
# Get latest run
run_id = client.list_runs()[-1]
run = client.get_run_by_id(run_id)
```

## Run outputs

After the run is done, the `output` property will contain a dictionary with the outputs. The
dictionary keys are the workflow sinks. The outputs will be `DataVibe`-like objects, that contain
metadata about the outputs, and references to the data files as assets. See the example below

```python
>>> run.status
<RunStatus.done: 'done'>
>>> run.output.keys()
dict_keys(['raster'])
>>> out = run.output["raster"]
>>> out
[Raster(id='3339a6f3-1800-4c1a-9edd-5b791734f240', time_range=(datetime.datetime(2020, 1, 1, 0, 0, tzinfo=datetime.timezone.utc), datetime.datetime(2022, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)), bbox=(-122.142363, 47.667801, -122.106146, 47.681775), geometry={'type': 'Polygon', 'coordinates': [[[-122.106146, 47.681775], [-122.106146, 47.667801], [-122.142363, 47.667801], [-122.142363, 47.681775], [-122.106146, 47.681775]]]}, assets=[AssetVibe(type='image/tiff', id='baa45c36-648b-4a03-9f3f-51ec9ff9d061', path_or_url='/data/cache/farmvibes-ai/data/assets/baa45c36-648b-4a03-9f3f-51ec9ff9d061/baa45c36-648b-4a03-9f3f-51ec9ff9d061.tif', _is_local=True, _local_path='/data/cache/farmvibes-ai/data/assets/baa45c36-648b-4a03-9f3f-51ec9ff9d061/baa45c36-648b-4a03-9f3f-51ec9ff9d061.tif')], bands={'red': 0, 'blue': 1, 'green': 2})]
>>> out[0].assets  # Check list of assets
[AssetVibe(type='image/tiff', id='baa45c36-648b-4a03-9f3f-51ec9ff9d061', path_or_url='/data/cache/farmvibes-ai/data/assets/baa45c36-648b-4a03-9f3f-51ec9ff9d061/baa45c36-648b-4a03-9f3f-51ec9ff9d061.tif', _is_local=True, _local_path='/data/cache/farmvibes-ai/data/assets/baa45c36-648b-4a03-9f3f-51ec9ff9d061/baa45c36-648b-4a03-9f3f-51ec9ff9d061.tif')]
>>> out[0].raster_asset.url  # Get reference for the generated tiff file
'file:///data/cache/farmvibes-ai/data/assets/baa45c36-648b-4a03-9f3f-51ec9ff9d061/baa45c36-648b-4a03-9f3f-51ec9ff9d061.tif'
```

## Submitting custom workflows

Instead of submitting a run with a built-in workflow name, it is also possible to send a workflow definition for a custom workflow.
The workflow definition is a dictionary that defines sources (inputs), sinks (outputs), parameters, tasks, and edges.
See [the workflow documentation](WORKFLOWS.md) for more information on how the the structure and syntax of workflow definitions.
Consider a case where we want to obtain NDVI rasters from Sentinel-2 imagery.
We can do this by composing a workflow that downloads and preprocesses Sentinel-2 data,
and a workflow that computes NDVI indices. The workflow definition is shown below:

```yaml
name: custom_ndvi_workflow
sources:
  user_input:
    - s2.user_input
sinks:
  ndvi: ndvi.index_raster
parameters:
  pc_key:
tasks:
  s2:
    workflow: data_ingestion/sentinel2/preprocess_s2
    parameters:
      # This parameter will have it's value filled by the workflow parameter
      pc_key: "@from(pc_key)"
  ndvi:
    workflow: data_processing/index/index
    parameters:
      # Set the index to NDVI
      index: ndvi
edges:
  - origin: s2.raster
    destination:
      - ndvi.raster
```

To submit the workflow, send the dictionary instead of a workflow name:

```python
import yaml

with open("custom_ndvi_workflow.yaml") as f:
    custom_wf = yaml.safe_load(f)
run = client.run(custom_wf, "Custom run name", geometry=my_geometry, time_range=my_time_range)
```

The custom workflow can be a composition of any of the available workflows.
It is not possible to use a custom workflow as a task to another workflow.
