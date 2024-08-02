# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Interact with NASA's EarthData platform's API
"""

import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from requests.exceptions import HTTPError
from shapely import geometry as shpg
from shapely import ops as shpo
from shapely.geometry.base import BaseGeometry

from vibe_core.data.core_types import BBox


def format_geometry(geometry: Union[shpg.Polygon, shpg.MultiPolygon]) -> List[str]:
    def format_poly(poly: shpg.Polygon):
        # Make sure it is a 2D geometry, and buffer 0 to make it more well-behaved
        # Orient to have the exterior go counter-clockwise
        poly = shpg.polygon.orient(shpo.transform(lambda *args: args[:2], poly.buffer(0)))
        assert poly.exterior is not None
        return ",".join(str(c) for p in poly.exterior.coords for c in p)

    if isinstance(geometry, shpg.MultiPolygon):
        geoms = [format_poly(p) for p in geometry.geoms]
    else:
        geoms = [format_poly(geometry)]
    return geoms


class EarthDataAPI:
    url: str = "https://cmr.earthdata.nasa.gov/search/granules.json"
    concept_ids: Dict[str, str] = {
        "GEDI01_B.002": "C1908344278-LPDAAC_ECS",
        "GEDI02_A.002": "C1908348134-LPDAAC_ECS",
        "GEDI02_B.002": "C1908350066-LPDAAC_ECS",
    }
    provider: str = "LPDAAC_ECS"
    page_size: int = 2000
    max_items: int = 1_000_000

    def __init__(self, processing_level: str):
        self.processing_level = processing_level
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _get_payload(
        self,
        *,
        geometry: Optional[BaseGeometry],
        bbox: Optional[BBox],
        time_range: Optional[Tuple[datetime, datetime]],
        id: Optional[str],
    ):
        """
        Build query parameters
        """
        # Format time range
        payload: Dict[str, Any] = {
            "provider": self.provider,
            "concept_id": self.concept_ids[self.processing_level],
            "page_size": self.page_size,
        }
        if time_range is not None:
            fmt_tr = ",".join(
                (t.astimezone().isoformat().replace("+00:00", "Z") for t in time_range)
            )
            payload["temporal"] = fmt_tr
        # Format spatial query
        if geometry is not None:
            assert isinstance(geometry, (shpg.Polygon, shpg.MultiPolygon))
            # Set option to get data that intersects with any of the geometries
            payload.update({"polygon[]": format_geometry(geometry), "options[polygon][or]": "true"})
        if bbox is not None:
            payload["bounding_box"] = ",".join(str(i) for i in bbox)
        if id is not None:
            payload["producer_granule_id"] = id
        return payload

    def query(
        self,
        *,
        geometry: Optional[BaseGeometry] = None,
        bbox: Optional[BBox] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        items = []
        max_pages = math.ceil(self.max_items / self.page_size)
        # Go to max_pages + 1 in case we have the maximum number of items possible
        # In practice we'll accept up to page_size - 1 extra items
        for page_num in range(1, max_pages + 2):
            payload = self._get_payload(geometry=geometry, bbox=bbox, time_range=time_range, id=id)
            payload["pageNum"] = page_num
            response = requests.post(self.url, data=payload)
            try:
                response.raise_for_status()
            except HTTPError as e:
                error_message = response.text
                msg = f"{e}. {error_message}"
                raise HTTPError(msg, response=e.response)
            page_items = response.json()["feed"]["entry"]
            num_items = len(page_items)
            self.logger.debug(f"Found {num_items} granules on page {page_num}")
            items.extend(page_items)
            if num_items < self.page_size:
                return items
        raise RuntimeError("Went through the maximum number of pages and did not return")
