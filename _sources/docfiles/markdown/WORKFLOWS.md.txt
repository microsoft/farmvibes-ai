# Workflows

A workflow defines a set of tasks the cluster should run, as well as the routing between inputs and outputs of each task.
We group FarmVibes.AI workflows in the following categories:

- **Data Ingestion**: workflows that download and preprocess data from a particular source, preparing data to be the starting point for most of the other workflows in the platform.
This includes raw data sources (e.g., Sentinel 1 and 2, LandSat, CropDataLayer) as well as the SpaceEye cloud-removal model;
- **Data Processing**: workflows that transform data into different data types (e.g., computing NDVI/MSAVI/Methane indexes, aggregating mean/max/min statistics of rasters, timeseries aggregation);
- **FarmAI**:  composed workflows (data ingestion + processing) whose outputs enable FarmAI scenarios (e.g., predicting conservation practices, estimating soil carbon sequestration, identifying methane leakage);
- **ML**: machine learning-related workflows to train, evaluate, and infer models within the FarmVibes.AI platform (e.g., dataset creation, inference);

For a list of all available workflows within the FarmVibes.AI platform, please
refer to [the workflow list](./WORKFLOW_LIST.md).

## Building a workflow

A workflow is defined by five main parts:

  1. Sources: inputs to the workflow. It is a dictionary that maps a named workflow input to a list of task input ports.
  2. Sinks: outputs of the workflow. It is a dictionary that maps a named workflow output to a task output port.
  The output of a run request will be a dictionary whose keys are the defined output names.
  3. Parameters: workflow parameters that can be changed at each run. A dictionary of named parameters and (optionally) default values.
  For more information about workflow parameters, see [Workflow Parameters](#workflow-parameters).
  4. Tasks: which tasks (other workflows or ops) the cluster should run.
  It is a dictionary that has a task name as key, and the task definition as value.
  5. Edges: connections between task output and input ports.
  A list of dictionaries containing the name of the output port (`origin`) and a list of input ports (`destination`).

The main way to define a workflow is via a yaml file. An example definition and an equivalent diagram are show below:

```yaml
name: example_workflow
sources:
  first_input:
    - first_task.input_port
    - second_task.port_input
  second_input:
    - third_task.input
sinks:
  first_output: first_task.output_port
  second_output: third_task.output
parameters:
  unused_param: 0
tasks:
  first_task:
    op: op_name  # this task is an op
  second_task:
    # this task is another workflow, that contains its own set of tasks
    workflow: another_workflow
  third_task:
    op: another_op
edges:
  - origin: second_task.some_output_port
    destination:
      - third_task.second_input
```

```{mermaid}
graph TD
  inp1>first_input]
  inp2>second_input]
  out1>first_output]
  out2>second_output]

  op1{{first_task}}
  wf1((second_task))
  op2{{third_task}}

  inp1 -- "input_port"--> op1
  inp1 -- "port_input"--> wf1
  inp2 -- "input"--> op2
  op1 -- "output_port" --> out1
  wf1 -- "some_output_port → second_input" --> op2
  op2 --> out2
```

## Workflow parameters

Workflow parameters can be used to customize workflow behavior by changing task parameters.
The workflow parameters are defined as a dictionary, with the key being the parameter name, and the value being an optional default parameter value.
If no default value is provided (`null`), the default value defined in the task is used.

Defining the workflow parameters is only the first step.
The second step is to map a workflow parameter into one or multiple task parameters.
This can be done by overriding the task parameter using `@from(wf_param)`, where `wf_param` is the name of the workflow parameter.

Consider both op and workflow defined below:

```yaml
name: example_op
inputs:
  input: DataVibe
outputs:
  output: DataVibe
parameters:
  op_param1: 1
  op_param2: 10
  unexposed_param: default
entrypoint:
  file: some_file.py
  callback_builder: callback_builder
```

```yaml
name: parameterizable_workflow
sources:
  input:
    - task.input
sinks:
  output:
    - task.output
parameters:
  # Default value will be 0
  wf_param1: 0
  # Default value depends on the value defined on the task
  wf_param2:
tasks:
  task:
    op: example_op
    parameters:
      op_param1: "@from(wf_param1)"
      op_param2: "@from(wf_param2)"
edges:
```

In this case, we define two workflow parameters, and map each of them to a different parameter in the task (note that one workflow parameter can be mapped to multiple task parameters).
Since we define a default value for `wf_param1`, a run submission without workflow parameter overrides will use 0 as default value.
In the case of `wf_param2`, since the default value is absent, we delegate the default value definition to the task (which is 10 is this case).
There is also `unexposed_param`, which is a task parameter that is not mapped to any workflow parameter, which means it cannot be changed using a workflow parameter override on submission.
It is up to the workflow creator to define which task parameters should be exposed at the workflow level.

## Workflow composition

Parameter override works the same regardless of the task being a workflow or an op.
If the task is a workflow, the parameter will override the inner workflow parameter value, and that value will be routed to its own tasks as defined in its workflow definition.
For example, if a workflow uses `parametetrizable_workflow` as a task and overrides `wf_param1` with no default value, `example_op` would run with default value of `op_param1 = 0`, as defined in `wf_param1`.
If `wf_param2` was overriden with no default value, `example_op` would run with default value `op_param2 = 10` as defined in the op, since `parameterizable_workflow` itself also does not define a default value for `wf_param2`.
It is not possible to override the parameter of an op inside a workflow if that workflow does not expose that parameter as a workflow parameter, so `unexposed_parameter` could not be changed by a workflow that uses `parameterizable_workflow` as a task.
