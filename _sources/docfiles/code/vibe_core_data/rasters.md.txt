# Rasters

The classes and methods defined by `vibe_core.data.rasters` module handle raster data related to remote sensing products and other geospatial data types. The base class in the module is `Raster`, which provides properties and methods for working with raster data, such as accessing raster and visualization assets. The module also includes specialized classes for different derived types, such as `DemRaster`, `NaipRaster`, `LandsatRaster`, and `GNATSGORaster`, which inherit from both the `Raster` class and their respective product metadata classes from `vibe_core.data.products`.

Additionally, the module provides classes for handling raster sequences (`RasterSequence`), raster chunks (`RasterChunk`), categorical rasters (`CategoricalRaster`), among others.

## Hierarchy

```{eval-rst}
.. raw:: html
   :file: ../../markdown/data_types_diagram/rasters_hierarchy.md
```

## Documentation

```{eval-rst}
.. automodule:: vibe_core.data.rasters
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
```
