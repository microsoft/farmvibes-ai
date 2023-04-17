# FarmVibes.AI

Welcome to the documentation of FarmVibes.AI, a platform for developing multi-modal geospatial machine learning (ML) models for agriculture and sustainability. The goal of the platform is to help users obtain insights by fusing multiple geospatial and spatiotemporal datasets.

The platform includes a set of data ingestion and pre-processing workflows that help users prepare data for fusion models tailored towards agriculture. The platform comes with many dataset downloaders, including satellite imagery from Sentinel 1 and 2, US Cropland Data, USGS Elevation maps, NAIP imagery, NOAA weather data, and private weather data from Ambient Weather. Users can also bring in any rasterized datasets they want to make them fusion-ready for FarmVibes.AI, such as drone imagery or other satellite imagery. We also provide multiple data processing workflows that can be used on top of downloaded data, such as, index computation and data summarization.

Currently, FarmVibes.AI is open-sourcing the local FarmVibes.AI cluster, which uses pre-built operators and workflows and runs them locally on the user's data science machine. This means that any data generated is persisted locally on the user's machine. The user can interact with the local FarmVibes.AI cluster via a REST API (in localhost) or a local Python client (inside a Jupyter Notebook, for example).

This documentation is a work in progress and will be updated regularly. If you have any questions or find any issues while running the code, please make sure to open an issue or a discussion on our [GitHub repository](https://github.com/microsoft/farmvibes-ai).

## Quickstart & Useful Links

Please refer to the the [Quickstart guide](./docfiles/markdown/QUICKSTART.md) for information on how to install and get started using the platform. If you prefer to setup a dedicated Azure Virtual Machine to run FarmVibes.AI, you can find detailed instructions in the [VM setup guide](./docfiles/markdown/VM-SETUP.md).

Additionally, the following user guides and links may be helpful:

- [FarmVibes.AI client user guide](./docfiles/markdown/CLIENT.md)
- [Workflows user guide](./docfiles/markdown/WORKFLOWS.md)
- [Troubleshooting](./docfiles/markdown/TROUBLESHOOTING.md)

<!-- ------------------------------------------------------------------------ -->

<!-- ## Documentation

This documentation details the python client and the main data types necessary to interact with the platform. We also provide the description and yaml files for the workflows that are currently available in FarmVibes.AI, showing how to compose and run user-defined workflows. Finally, we also document the [Microsoft Azure Data Manager for Agriculture (ADMaG)](https://learn.microsoft.com/en-us/azure/data-manager-for-agri/overview-azure-data-manager-for-agriculture) client, which can be used to interact with the ADMaG platform. -->

```{eval-rst}
.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: User Guides & Tutorials:

   docfiles/markdown/QUICKSTART
   docfiles/markdown/CLIENT
   docfiles/markdown/WORKFLOWS
   docfiles/markdown/NOTEBOOK_LIST
   docfiles/markdown/SECRETS
   docfiles/markdown/TROUBLESHOOTING
```

```{eval-rst}
.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Main Components:

   docfiles/code/vibe_core_client/client
   docfiles/code/vibe_core_data/index_data_types
   docfiles/markdown/WORKFLOW_LIST
```

```{eval-rst}
.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Supporting Components:

   docfiles/code/vibe_core_client/admag_client
   docfiles/code/vibe_core_client/datamodel
   docfiles/code/vibe_core_client/additional_modules/index
```

## Indices and tables

You can also look for classes and methods by name using the following indices:

- {ref}`genindex`
- {ref}`modindex`
