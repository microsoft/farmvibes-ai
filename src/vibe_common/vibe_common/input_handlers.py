from datetime import datetime
from typing import Any, Dict

from vibe_core.data import DataVibe, StacConverter, gen_hash_id

# Checking geojson dict and extracting geometry
VALID_GEOMETRIES = ["Polygon", "MultiPolygon"]
INVALID_GEOMETRIES = [
    "Point",
    "LineString",
    "MultiPoint",
    "MultiLineString",
    "GeometryCollection",
]


def handle_non_collection(
    geojson_dict: Dict[str, Any], start_date: datetime, end_date: datetime
) -> Dict[str, Any]:
    geotype = geojson_dict["type"]

    if geotype == "Feature":
        geometry = geojson_dict["geometry"]
    elif geotype in VALID_GEOMETRIES:
        geometry = geojson_dict
    elif geotype == "FeatureCollection":
        raise ValueError("Feature collection not supported here.")
    elif geotype in INVALID_GEOMETRIES:
        raise ValueError(
            f"Invalid geometry {geotype}. Input geometry must be Polygon or MultiPolygon."
        )
    else:
        raise ValueError(f"Invalid geojson type {geotype}.")

    converter = StacConverter()
    time_range = (start_date, end_date)
    data = DataVibe(
        id=gen_hash_id("input", geometry, time_range),
        time_range=time_range,
        geometry=geometry,
        assets=[],
    )
    stac_item = converter.to_stac_item(data)

    return stac_item.to_dict(include_self_link=False)


def gen_stac_item_from_bounds(
    geojson_dict: Dict[str, Any], start_date: datetime, end_date: datetime
) -> Dict[str, Any]:
    geotype = geojson_dict["type"]

    if geotype == "FeatureCollection":
        if len(geoms := geojson_dict["features"]) > 1:
            raise ValueError(
                f"Only one feature is currently supported as input to a workflow, found "
                f"{len(geoms)} features in feature collection"
            )
        return handle_non_collection(geoms[0], start_date, end_date)

    return handle_non_collection(geojson_dict, start_date, end_date)
