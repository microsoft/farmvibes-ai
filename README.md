# FarmVibes.AI: Multi-Modal GeoSpatial ML Models for Agriculture and Sustainability

With FarmVibes.AI, you can develop rich geospatial insights for agriculture and sustainability.

Build models that fuse multiple geospatial and spatiotemporal datasets to obtain insights (e.g.
estimate carbon footprint, understand growth rate, detect practices followed) that would be 
hard to obtain when these datasets are used in isolation. You can fuse together satellite imagery 
(RGB, SAR, multispectral), drone imagery, weather data, and more.

Fusing datasets this way helps generate more robust insights and unlocks new insights that are
otherwise not possible without fusion. This repo contains several fusion workflows (published and
shown to be key for agriculture related problems) that help you build robust remote sensing, earth
observation, and geospatial models with focus on agriculture/farming with ease. Our main focus right
now is agriculture and sustainability, which the models are optimized for. However, the framework itself is generic
enough to help you build models for other domains.

## FarmVibes.AI Primer

There are three main pieces to FarmVibes.AI. The first one consists of data ingestion and 
pre-processing workflows to help prepare data for fusion models tailored towards agriculture. 
Additionally, we provide model training notebook examples that not only allow the configuration 
of pre-processing of data but also allow tuning existing models with ease. Finally, a compute 
engine that supports data ingestion as well as adjusting existing and creating novel workflows 
with the tuned model.

### FarmVibes.AI Fusion-Ready Dataset Preparation

In this step, you can select the datasets that you would like to fuse for building the insights.
FarmVibes.AI comes with many dataset downloaders. These include satellite imagery from Sentinel 1
and 2, US Cropland Data, USGS Elevation maps, NAIP imagery, NOAA weather data, private weather data
from Ambient Weather. Additionally, you can also bring in any rasterized datasets that you
want to make them fusion-ready for FarmVibes.AI (e.g. drone imagery or other satellite imagery) and, in 
the future, custom sensor data (such as weather sensors).

The key technique in FarmVibes.AI is to use as input for ML models data that goes much beyond 
types, space and time from where the labels are located. For example, when detecting grain silos
from satellite imagery (labeled only in optical imagery), it is better to rely on optical as well as
elevation and radar bands. In this scenario, it is also important to combine multiple data modalities with other known agriculture infrastructure entities. Likewise, it is also
important to use as input the images of a given silo across various times of the year to help
generate a more robust model. Including information from many data streams, while also incorporating 
historical data from nearby or similar locations  has been shown to improve
robustness of geospatial models (especially for yield, growth, and crop classification problems).
FarmVibes.AI generates such input data for models with ease based on parameters that can be
specified.

FarmVibes.AI enables a data scientist to massage and/or tune the datasets to their preferences. The
tuning is enabled via a configurable workflow which is specified as a directed acyclic graph of data
downloading workflows and data preparation workflows. The preparation operators help create the
inputs (e.g. fused pandas arrays or tensors containing all raw data) to training and inference
modules.

### FarmVibes.AI Model Sample Notebook Library

The next step in FarmVibes.AI involves using the inbuilt notebooks to tune the models to achieve a
level of accuracy for the parts of the world or seasons that you are focusing on. The library
includes notebooks for  detecting practices (e.g. harvest date detection), estimating climate impact
(both seasonal carbon footprint and long term sustainability), micro climate prediction, and crop
identification. 

FarmVibes.AI comes with these notebooks to help you get started to train fusion models to combine 
the geospatial datasets into robust insights tailored for your needs. The users can tune the model to
 a desired performance and publish the model to FarmVibes.AI. The model then shows up to be used later in an inference engine that can be employed for other parts of the world, other dates, or more.

### FarmVibes.AI Inference Engine

The final stage in FarmVibes.AI is to combine the data connectors, pre-processing, and the model
pieces together into a robust inference workflow. The generated workflow can then be used for
performing inference in an area of interest and time range that can be passed as inputs to the
workflow. FarmVibes.AI can be configured such that it then runs the inference for the time range and
updates the results whenever upstream data is updated (e.g. new satellite imagery or sensor data is
added). You do this by creating a workflow that is composed of fused data preparation and fusion
model workflows.

## Operation Mode

Currently, we are open-sourcing the local FarmVibes.AI cluster, that uses pre-build operators and
workflows and runs them locally on your data science machine. This means that any data generated is
persisted locally in your machine. The actual workflows are provided via Docker images, with description
in the [workflow list documentation](./docs/source/docfiles/markdown/WORKFLOW_LIST.md).

The user can interact with the local FarmVibes.AI cluster via a REST API (in localhost) or a local
Python client (inside a Jupyter Notebook, for example).

## Installation

Please refer to the the [Quickstart guide](./docs/source/docfiles/markdown/QUICKSTART.md) for information on where to get started. If
you prefer to setup a dedicated Azure Virtual Machine to run FarmVibes.AI, you can find detailed
instructions [in the VM setup documentation](./docs/source/docfiles/markdown/VM-SETUP.md).

## Notebook Examples

In the folder `notebooks` there are several examples to serve as starting points and demonstrating
how FarmVibes.AI can be used to create Agriculture insights. Some of the available notebooks are:

* `helloworld`: a simple example on how to use the client to run a workflow and visualize the
response.
* `harvest_period`: showing how a NDVI time-series computed on top of Sentinel 2 data can
be obtained for a single field and planting season and used to estimate emergence and harvest dates.
* `carbon`: illustrating how to simulate different soil carbon estimates based on different
agriculture practices, leveraging the [COMET-Farm API](https://gitlab.com/comet-api/api-docs/-/tree/master/).
* `deepmc`: showing how one can build micro-climate forecasts from weather station data using the
[DeepMC model](https://spectrum.ieee.org/deepmc-weather-predicition). 
* `crop_segementation`: this
example shows how to train a crop identification model based on NDVI data computed on top of our
[SpaceEye](https://arxiv.org/abs/2106.08408) cloud-free image generation model. In this example, you 
can also then use the trained model in an inference workflow to obtain predictions in any area where 
we are able to generate SpaceEye imagery.

A complete list of the notebooks available and their description is available [here](./docs/source/docfiles/markdown/NOTEBOOK_LIST.md). 

## Documentation

In the `documentation` folder more detailed information about the different components can be found.
In particular: 
* [FARMVIBES_AI.md](./docs/source/docfiles/markdown/FARMVIBES_AI.md) describing how to setup and
manage the local cluster. 
* [WORKFLOWS.md](./docs/source/docfiles/markdown/WORKFLOWS.md) describing how workflows
can be written and how they function.
* [CLIENT.md](./docs/source/docfiles/markdown/CLIENT.md) documenting the
FarmVibes.AI client, which is the preferred way to run workflows and interact with the results. 
* [SECRETS.md](./docs/source/docfiles/markdown/SECRETS.md) describing how to manage and pass secrets to the cluster
(such as API keys), so that they will be available when running workflows. 
* [TROUBLESHOOTING.md](./docs/source/docfiles/markdown/TROUBLESHOOTING.md) in case you run into any issues.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
