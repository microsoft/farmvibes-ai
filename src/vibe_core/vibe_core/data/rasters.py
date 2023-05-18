from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from shapely import geometry as shpg

from .core_types import AssetVibe, DataSequence, DataVibe
from .products import DemProduct, GNATSGOProduct, LandsatProduct, NaipProduct

ChunkLimits = Tuple[int, int, int, int]
"""Type alias for chunk limits. Tuple of col_offset, row_offset, width, height."""


@dataclass
class Raster(DataVibe):
    """Represents raster data in FarmVibes.AI."""

    bands: Dict[str, int]
    """A dictionary with the name of each band and its index in the raster data."""

    def __post_init__(self):
        super().__post_init__()
        self.scale = 1
        self.offset = 0

    @property
    def raster_asset(self) -> AssetVibe:
        """Returns the raster asset from the list of assets.

        :raises ValueError: If the raster asset cannot be found in the asset list.

        :returns: The raster asset from the asset list.
        """
        raster_asset = [a for a in self.assets if (a.type is not None) and ("image/" in a.type)]
        if raster_asset:
            return raster_asset[0]
        raise ValueError(f"Could not find raster asset in asset list: {self.assets}")

    @property
    def visualization_asset(self) -> AssetVibe:
        """Returns the visualization asset from the asset list.

        :raises ValueError: If the visualization asset cannot be found in the asset list.

        :returns: The visualization asset from the asset list.
        """
        vis_asset = [a for a in self.assets if a.type == "application/json"]
        if vis_asset:
            return vis_asset[0]
        raise ValueError(f"Could not find visualization asset in asset list: {self.assets}")


@dataclass
class RasterSequence(DataSequence, Raster):
    """Represents a sequence of rasters"""

    def add_item(self, item: Raster):
        """Adds a raster to the sequence

        :param item: The raster to add to the sequence
        """
        self.add_asset(item.raster_asset, item.time_range, shpg.shape(item.geometry))


@dataclass
class RasterChunk(Raster):
    """Represents a chunk of a raster."""

    chunk_pos: Tuple[int, int]
    """The position of the chunk in the raster data, as a tuple of (column, row) indices."""
    num_chunks: Tuple[int, int]
    """The total number of chunks in the raster data, as a
    tuple of (number of columns, number of rows).
    """
    limits: ChunkLimits
    """The limits of the chunk in the raster data, as a :const:`ChunkLimits` object.
    These are indices, not coordinates.
    """
    write_rel_limits: ChunkLimits
    """The relative limits of the chunk in the raster data asset. These are non-overlapping
    indices that are used to write the chunk to the asset.
    """


@dataclass
class CategoricalRaster(Raster):
    """Represents a categorical raster."""

    categories: List[str]
    """The list of categories in the raster."""


@dataclass
class CloudRaster(Raster):
    """Represents a cloud raster."""

    bands: Dict[str, int] = field(init=False)
    """A dictionary with the name of each band and its index in the raster data."""

    def __post_init__(self):
        super().__post_init__()
        self.bands = {"cloud": 0}


@dataclass
class RasterIlluminance(DataVibe):
    """Represents illuminance values for bands of a raster."""

    illuminance: List[float]
    """The list of illuminance values for each band."""


@dataclass
class DemRaster(Raster, DemProduct):
    """Represents a DEM raster."""

    pass


@dataclass
class NaipRaster(Raster, NaipProduct):
    """Represents a NAIP raster."""

    pass


@dataclass
class LandsatRaster(LandsatProduct, Raster):
    """Represents a Landsat raster."""

    def __post_init__(self):
        super().__post_init__()
        self.scale = 2.75e-5
        self.offset = -0.2


@dataclass
class ModisRaster(Raster):
    """Represents a MODIS raster."""

    def __post_init__(self):
        super().__post_init__()
        self.scale = 1e-4


@dataclass
class GNATSGORaster(Raster, GNATSGOProduct):
    """Represents a gNATSGO raster of a specific variable."""

    variable: str
    """The variable represented in the raster."""
