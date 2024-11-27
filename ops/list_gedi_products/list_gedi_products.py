# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
from typing import Any, Dict, List

from dateutil.parser import parse as parse_date
from shapely import geometry as shpg

from vibe_core.data import DataVibe, GEDIProduct
from vibe_lib.earthdata import EarthDataAPI

LOGGER = logging.getLogger(__name__)


def parse_poly(poly_str: str) -> shpg.Polygon:
    coords = poly_str.split(" ")
    return shpg.Polygon([(float(c2), float(c1)) for c1, c2 in zip(coords[::2], coords[1::2])])


def convert_product(item: Dict[str, Any]) -> GEDIProduct:
    geoms = [parse_poly(pp) for p in item["polygons"] for pp in p]
    product_id = item["producer_granule_id"]
    if not geoms:
        raise RuntimeError(f"Failed to parse geometry from GEDI Product {product_id}")
    if len(geoms) > 1:
        geom = shpg.MultiPolygon(geoms)
    else:
        geom = geoms[0]
    time_range = tuple(parse_date(item[k]) for k in ("time_start", "time_end"))
    orbits = item["orbit_calculated_spatial_domains"][0]
    concept_id = item["collection_concept_id"]
    processing_level = [k for k, v in EarthDataAPI.concept_ids.items() if v == concept_id]
    if len(processing_level) == 0:
        raise RuntimeError(f"Failed to parse concept id {concept_id} from product {product_id}")
    processing_level = processing_level[0]
    return GEDIProduct(
        id=product_id,
        geometry=shpg.mapping(geom),
        time_range=time_range,
        product_name=product_id,
        start_orbit=int(orbits["start_orbit_number"]),
        stop_orbit=int(orbits["stop_orbit_number"]),
        processing_level=processing_level,
        assets=[],
    )


def callback_builder(processing_level: str):
    if processing_level not in EarthDataAPI.concept_ids:
        valid_levels = ", ".join([f"'{i}'" for i in EarthDataAPI.concept_ids])
        raise ValueError(f"Parameters processing_level must be one of {valid_levels}")

    def callback(input_data: DataVibe) -> Dict[str, List[GEDIProduct]]:
        api = EarthDataAPI(processing_level)
        geom = shpg.shape(input_data.geometry)
        time_range = input_data.time_range
        LOGGER.info(
            f"Querying EarthData API for {processing_level=}, "
            f"geometry={shpg.mapping(geom)}, {time_range=}"
        )
        items = api.query(geometry=geom, time_range=time_range)
        if not items:
            raise RuntimeError(
                f"Query returned no items for time range {time_range} "
                f"and geometry {shpg.mapping(geom)}"
            )
        LOGGER.info(f"EarthData API returned {len(items)} items. Converting to DataVibe")
        products = [convert_product(i) for i in items]
        return {"gedi_products": products}

    return callback
