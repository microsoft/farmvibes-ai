import hashlib
import logging
from collections import defaultdict
from datetime import timedelta
from typing import Dict, List, Tuple, cast

import fiona
import geopandas as gpd
from shapely import geometry as shpg
from shapely.geometry.base import BaseGeometry

from vibe_core.data import BBox, DataVibe, TimeRange
from vibe_core.data.sentinel import ListTileData, Tile2Sequence, TileData, TileSequenceData
from vibe_lib.spaceeye.dataset import get_read_intervals, get_write_intervals

LOGGER = logging.getLogger(__name__)
KML_DRIVER_NAMES = "kml KML libkml LIBKML".split()


def gen_sequence_id(
    items: ListTileData,
    geom: BaseGeometry,
    read_time_range: TimeRange,
    write_time_range: TimeRange,
):
    """Generate the id for a Tile Sequence, considering all rasters in the sequence"""
    id = hashlib.sha256(
        "".join(
            [i.id for i in items]
            + [geom.wkt]
            + [
                t.isoformat()
                for time_range in (read_time_range, write_time_range)
                for t in time_range
            ]
        ).encode()
    ).hexdigest()
    return id


def group_rasters(rasters: ListTileData, input_data: List[DataVibe], tile_dfs: gpd.GeoDataFrame):
    """Group rasters covering the same region (intersection between input geometry and a tile)"""
    sequences: Dict[Tuple[str, BBox], ListTileData] = defaultdict(list)
    sequences_geom: Dict[Tuple[str, BBox], BaseGeometry] = defaultdict()
    sequences_time_range: Dict[Tuple[str, BBox], TimeRange] = defaultdict()

    # Iterate over all rasters that cover the input geometries
    for item in rasters:
        tile_id = item.tile_id
        tile_geom = tile_dfs.loc[tile_dfs["Name"] == tile_id]["geometry"].iloc[0]  # type: ignore
        tile_start_date = item.time_range[0]

        # For now, we only consider a single geometry within input_data. In the future,
        # we might allow multiple geometries, so this already covers that.
        for input_geom in input_data:
            # We are interested in the intersection between tile geom and input geometry
            # for all tiles captured within the time range of the input geometry
            geom = shpg.shape(input_geom.geometry)
            start_date, end_date = input_geom.time_range

            if (start_date <= tile_start_date <= end_date) and geom.intersects(tile_geom):
                intersected_geom = geom.intersection(tile_geom)

                # Use tile id and bounding box of intersecting region as keys
                sequence_key = (item.tile_id, tuple(intersected_geom.bounds))
                sequences[sequence_key].append(item)
                sequences_geom[sequence_key] = intersected_geom
                sequences_time_range[sequence_key] = input_geom.time_range

    return sequences, sequences_geom, sequences_time_range


def make_tile_sequence(
    items: ListTileData,
    seq_geom: BaseGeometry,
    read_time_range: TimeRange,
    write_time_range: TimeRange,
    ref_item: TileData,
) -> TileSequenceData:
    """Create a TileSequenceData from the list of rasters and a sequence geometry"""
    # Make sure we are ordered by time make things consistent for the id hash
    sequence_type = Tile2Sequence[type(ref_item)]
    sorted_items = sorted(items, key=lambda x: x.time_range[0])

    # Generate sequence metadata
    sequence_id = gen_sequence_id(sorted_items, seq_geom, read_time_range, write_time_range)

    # Create sequence object
    sequence = sequence_type.clone_from(
        ref_item,
        id=sequence_id,
        assets=[],
        geometry=shpg.mapping(seq_geom),
        time_range=read_time_range,
        write_time_range=write_time_range,
        product_name="",
        orbit_number=-1,
        relative_orbit_number=-1,
        orbit_direction="",
        platform="",
    )

    for r in sorted_items:
        sequence.add_item(r)

    return sequence


def make_chip_sequences(
    items: ListTileData,
    seq_geom: BaseGeometry,
    seq_time_range: TimeRange,
    duration: int,
    step: int,
) -> List[TileSequenceData]:
    ref_item = items[0]
    time_length = (seq_time_range[1] - seq_time_range[0]).days + 1
    if time_length < duration:
        LOGGER.warning(f"Time length of {time_length} days is smaller than chip length {duration}")
        offset = (time_length - duration) // 2
        time_length = duration
    else:
        offset = 0

    read_intervals = list(zip(*get_read_intervals(time_length, duration, step, 0)))
    write_intervals = list(zip(*get_write_intervals(time_length, duration, step, 0)[0]))

    sequences = []
    for read_interval, write_interval in zip(read_intervals, write_intervals):
        start, end = (seq_time_range[0] + timedelta(days=int(i) + offset) for i in read_interval)
        interval_items = [i for i in items if start <= i.time_range[0] < end]
        if not interval_items:
            LOGGER.warning(
                f"Time interval {start.strftime('%Y-%m-%d')} - {end.strftime('%Y-%m-%d')} has no "
                f"available data of type {type(ref_item)} for tile_id={ref_item.tile_id}, "
                f"geometry={shpg.mapping(seq_geom)}"
            )
        write_dates = (
            seq_time_range[0] + timedelta(days=int(write_interval[0]) + offset),
            seq_time_range[0] + timedelta(days=int(write_interval[1]) + offset - 1),  # type: ignore
        )
        # Use end - 1 because our date range is closed at the end and our index range is not
        sequences.append(
            make_tile_sequence(
                interval_items,
                seq_geom,
                (start, end - timedelta(days=1)),
                write_dates,
                ref_item,
            )
        )

    return sequences


class CallbackBuilder:
    def __init__(self, tile_geometry: str, duration: int, overlap: float):
        self.tile_geometry = tile_geometry
        self.duration = duration
        if duration <= 0:
            raise ValueError(f"Duration must be larger than 0, found {duration}")
        if overlap <= 0 or overlap > 1:
            raise ValueError(f"Overlap value must be in range [0, 1), found {overlap}")
        self.overlap = overlap

    def __call__(self):
        def group_by_tile_geom(
            rasters: ListTileData, input_data: List[DataVibe]
        ) -> Dict[str, List[TileSequenceData]]:
            # List the tiles for which we have products
            tile_ids = set(p.tile_id for p in rasters)

            # Read tile geometry and filter for those that we have products
            # Make fiona read the file: https://gis.stackexchange.com/questions/114066/
            for driver in KML_DRIVER_NAMES:
                fiona.drvsupport.supported_drivers[driver] = "rw"  # type: ignore
            tile_dfs = gpd.read_file(self.tile_geometry)
            # Filter only tiles for which we have products
            tile_dfs = cast(
                gpd.GeoDataFrame,
                tile_dfs[tile_dfs["Name"].isin(tile_ids)],  # type: ignore
            )

            # Group rasters by tile_id and geometry
            sequences, sequences_geom, sequences_time_range = group_rasters(
                rasters, input_data, tile_dfs
            )

            # Create TileSequenceData for each group
            step = int(self.duration * self.overlap)
            grouped_sequences = [
                group
                for k in sequences.keys()
                for group in make_chip_sequences(
                    sequences[k],
                    sequences_geom[k],
                    sequences_time_range[k],
                    self.duration,
                    step,
                )
            ]

            return {"tile_sequences": grouped_sequences}

        return group_by_tile_geom
