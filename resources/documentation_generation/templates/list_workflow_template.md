# Workflow List

We group FarmVibes.AI workflows in the following categories:

- **Data Ingestion**: workflows that download and preprocess data from a particular source, preparing data to be the starting point for most of the other workflows in the platform.
This includes raw data sources (e.g., Sentinel 1 and 2, LandSat, CropDataLayer) as well as the SpaceEye cloud-removal model;
- **Data Processing**: workflows that transform data into different data types (e.g., computing NDVI/MSAVI/Methane indexes, aggregating mean/max/min statistics of rasters, timeseries aggregation);
- **FarmAI**:  composed workflows (data ingestion + processing) whose outputs enable FarmAI scenarios (e.g., predicting conservation practices, estimating soil carbon sequestration, identifying methane leakage);
- **ForestAI**: composed workflows (data ingestion + processing) whose outputs enable ForestAI scenarios (e.g., detecting forest change, estimating forest extent);
- **ML**: machine learning-related workflows to train, evaluate, and infer models within the FarmVibes.AI platform (e.g., dataset creation, inference);

Below is a list of all available workflows within the FarmVibes.AI platform. For each of them, we provide a brief description and a link to the corresponding documentation page.

---------

{% for elem in data_source -%}

## {{elem.category}}

{% for wf in elem.wf_list %}- [`{{wf.name}}` 📄]({{wf.markdown_link}}): {{wf.description}}

{% endfor %}
{% endfor %}
