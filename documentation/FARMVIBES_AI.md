# FarmVibes.AI overview

In a hurry? If you want to install/run FarmVibes.AI right away, please refer to
the [quickstart guide](../QUICKSTART.md) for a fast introduction on how to install
and run FarmVibes.AI.

FarmVibes.AI provides a modular and scalable platform for processing geospatial data
at a large scale using reusable components. The platform is modular in the sense
that the computation is operation-driven and subdivided spatially and
temporally.  FarmVibes.AI automatically splits geospatial workflow computations
into multiple parallel chunks and reuse the previously computed results for
certain regions and timestamps.

FarmVibes.AI workflows calls are idempotent, different execution requests with
the same input and parameters always result in identical output.  Therefore, we
implemented a geospatial caching system to accelerate the computation of
previously executed workflows for the same geospatial region in the same
time window. This is particularly useful for data ingestion workflows that tend
to reevaluate the same area over and over again using different analytical
approaches. For instance, assume you are interested in performing a yield
forecast for a given crop in a specific time window and use temperature as
input.

The proposed platform splits the computation and ingestion of temperature data
into small chunks for this region and caches this data. Whenever we need to
perform another evaluation for the same area/time windows that has temperature
as input (for instance, wind prediction) FarmVibes.AI will use cached data to
accelerate the evaluation.

Workflows are computational graphs with typed edges with nodes representing a
processing step and edges representing the flow of inputs/outputs between each
atomic computation. Each computing node can receive a single item or a list of
items as input, where each item list has the same type. FarmVibes.AI leverages
that to parallelize the computation for each list element, accelerating even
further the evaluation performance. All this process is automatic and do not
require user intervention.

Given its parallel nature, FarmVibes.AI is a  project specially
designed to leverage cloud elasticity and scalability characteristics. In this sense,
the proposed project has the following components.

## FarmVibes.AI cluster

FarmVibes.AI cluster  is a kubernetes-based set of computing pods capable of
running multiple workflows in parallel. The cluster has four major components:

1. **Rest-api (Server).** A webserver that exposes a REST API so users can
call workflows, track workflow execution and retrieve results.

2. **Orchestrator.** This component manages workflow execution, transmitting
requests to workers and updating workflow status.

3. **Worker.** A scalable component responsible for the actual workflow
operation computation. Instead of running the whole workflow at once, it
computes the atomic chunks processed by the user.

4. **Cache.** This component sits between the orchestrator and
workers, it checks if an operation was previously executed and returns
cached results to the orchestrator.

To check how to configure and install the FarmVibes.AI cluster, please
issue the following command in the project root folder.

```bash
bash farmvibes-ai.sh -h
```

## FarmVibes.AI REST API

FarmVibes.AI provides a REST API to manage workflows execution. Assuming there
is cluster running, then the url `http://<cluster_addr>:<port>/v0/docs/` should
provide the REST API documentation (e.g., `http://192.168.49.2:30000/v0/docs`).

## FarmVibes.AI Python Client

Besides the REST API, we also provide a python client that abstracts the
communication with the cluster (please check [python client
documentation](./CLIENT.MD)).

## FarmVibes.AI workflow documentation

Dynamically generated workflow documentation can be accessed via python client as follows:

```python
>>> from vibe_core.client import get_default_vibe_client
>>> client = get_default_vibe_client()
>>> client.document_workflow("data_ingestion/spaceeye/spaceeye")
```

Please refer to the [python client documentation](./CLIENT.MD) to see how to get
a list of available workflows.
