# Core Types

The `vibe_core.data.core_types` provides a set of foundation classes and utilities to represent and manipulate various types of data used in the FarmVibes.AI. The `BaseVibe` class is the base class for all FarmVibes.AI types, and provides a common interface for data access and manipulation. The `DataVibe` class represents a data object in FarmVibes.AI, and includes properties such as a unique identifier, a time range, a bounding box, and a geometry. Other classes, such as `TimeSeries`, `DataSummaryStatistics`, `DataSequence`, inherit from `DataVibe` and provide additional functionality for specific data types.

The module is designed to handle a wide range of data types, including geospatial data, time series data, and more. It also provides utility functions and classes to help with tasks such as generating unique identifiers, validating data types, and parsing type specifications.

## Hierarchy

```{eval-rst}
.. raw:: html
   :file: ../../markdown/data_types_diagram/core_types_hierarchy.md
```

## Documentation

```{eval-rst}
.. automodule:: vibe_core.data.core_types
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
```
