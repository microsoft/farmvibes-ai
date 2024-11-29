"""Data types, constants, and supporting functions for manipulating rasters in FarmVibes.AI."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from shapely import geometry as shpg

from .core_types import AssetVibe, BBox, ChipWindow, DataSequence, DataVibe
from .products import DemProduct, GNATSGOProduct, LandsatProduct, NaipProduct

ChunkLimits = Tuple[int, int, int, int]
"""Type alias for chunk limits. Tuple of col_offset, row_offset, width, height."""

RASTER_ASSET_MIME = ["image/", "application/x-grib", "application/grib"]


@dataclass
class Raster(DataVibe):
    """Represent raster data in FarmVibes.AI."""

    bands: Dict[str, int]
    """A dictionary with the name of each band and its index in the raster data."""

    def __post_init__(self):
        super().__post_init__()
        self.scale = 1
        self.offset = 0

    @property
    def raster_asset(self) -> AssetVibe:
        """Return the raster asset from the list of assets.

        Returns:
            The raster asset from the asset list.

        Raises:
            ValueError: If the raster asset cannot be found in the asset list.
        """
        raster_asset = [
            a
            for a in self.assets
            if (a.type is not None) and any([(mime in a.type) for mime in RASTER_ASSET_MIME])
        ]
        if raster_asset:
            return raster_asset[0]
        raise ValueError(f"Could not find raster asset in asset list: {self.assets}")

    @property
    def visualization_asset(self) -> AssetVibe:
        """Return the visualization asset from the asset list.

        Returns:
            The visualization asset from the asset list.

        Raises:
            ValueError: If the visualization asset cannot be found in the asset list.
        """
        vis_asset = [a for a in self.assets if a.type == "application/json"]
        if vis_asset:
            return vis_asset[0]
        raise ValueError(f"Could not find visualization asset in asset list: {self.assets}")


@dataclass
class RasterSequence(DataSequence, Raster):
    """Represent a sequence of rasters."""

    def add_item(self, item: Raster):
        """Add a raster to the sequence.

        Args:
            item: The raster to add to the sequence.
        """
        self.add_asset(item.raster_asset, item.time_range, shpg.shape(item.geometry))


@dataclass
class RasterChunk(Raster):
    """Represent a chunk of a raster."""

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
    """Represent a categorical raster."""

    categories: List[str]
    """The list of categories in the raster."""


@dataclass
class CloudRaster(Raster):
    """Represent a cloud raster."""

    bands: Dict[str, int] = field(init=False)
    """A dictionary with the name of each band and its index in the raster data."""

    def __post_init__(self):
        super().__post_init__()
        self.bands = {"cloud": 0}


@dataclass
class RasterIlluminance(DataVibe):
    """Represent illuminance values for bands of a raster."""

    illuminance: List[float]
    """The list of illuminance values for each band."""


@dataclass
class DemRaster(Raster, DemProduct):
    """Represent a DEM raster."""

    pass


@dataclass
class NaipRaster(Raster, NaipProduct):
    """Represent a NAIP raster."""

    pass


@dataclass
class LandsatRaster(LandsatProduct, Raster):
    """Represent a Landsat raster."""

    def __post_init__(self):
        super().__post_init__()
        self.scale = 2.75e-5
        self.offset = -0.2


@dataclass
class ModisRaster(Raster):
    """Represent a MODIS raster."""

    def __post_init__(self):
        super().__post_init__()
        self.scale = 1e-4


@dataclass
class GNATSGORaster(Raster, GNATSGOProduct):
    """Represent a gNATSGO raster of a specific variable."""

    variable: str
    """The variable represented in the raster."""


@dataclass
class SamMaskRaster(CategoricalRaster):
    """Represent a raster with Segment Anything Model (SAM) masks.

    Each asset in the raster contains a mask obtained with SAM.
    """

    mask_score: List[float]
    """The list of SAM quality scores for each mask in the assets."""

    mask_bbox: List[BBox]
    """The list of bounding boxes for each mask in the assets."""

    chip_window: ChipWindow
    """The chip window (col_offset, row_offset, width, height) covered by this raster."""
