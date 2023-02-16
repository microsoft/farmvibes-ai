from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from shapely import geometry as shpg

from .core_types import AssetVibe, DataSequence, DataVibe
from .products import DemProduct, GNATSGOProduct, LandsatProduct, NaipProduct

# col_offset, row_offset, width, height
ChunkLimits = Tuple[int, int, int, int]


@dataclass
class Raster(DataVibe):
    bands: Dict[str, int]

    def __post_init__(self):
        super().__post_init__()
        self.quantification_value = 1

    @property
    def raster_asset(self) -> AssetVibe:
        raster_asset = [a for a in self.assets if (a.type is not None) and ("image/" in a.type)]
        if raster_asset:
            return raster_asset[0]
        raise ValueError(f"Could not find raster asset in asset list: {self.assets}")

    @property
    def visualization_asset(self) -> AssetVibe:
        vis_asset = [a for a in self.assets if a.type == "application/json"]
        if vis_asset:
            return vis_asset[0]
        raise ValueError(f"Could not find visualization asset in asset list: {self.assets}")


@dataclass
class RasterSequence(DataSequence, Raster):
    def add_item(self, item: Raster):
        self.add_asset(item.raster_asset, item.time_range, shpg.shape(item.geometry))


@dataclass
class RasterChunk(Raster):
    chunk_pos: Tuple[int, int]
    num_chunks: Tuple[int, int]

    # limits are indices not coordinates, coordinates of the chunk is
    # stored in geometry
    limits: ChunkLimits  # [col_off, row_off, width, height]

    # non-overlapping indices
    write_rel_limits: ChunkLimits


@dataclass
class CategoricalRaster(Raster):
    categories: List[str]


@dataclass
class CloudRaster(Raster):
    bands: Dict[str, int] = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.bands = {"cloud": 0}


@dataclass
class RasterIlluminance(DataVibe):
    illuminance: List[float]


@dataclass
class DemRaster(Raster, DemProduct):
    """
    DEM product downloaded with specific resolution. We might want to unify the raster classes
    in the future.
    """

    pass


@dataclass
class NaipRaster(Raster, NaipProduct):
    pass


@dataclass
class LandsatRaster(LandsatProduct, Raster):
    pass


@dataclass
class GNATSGORaster(Raster, GNATSGOProduct):
    variable: str
