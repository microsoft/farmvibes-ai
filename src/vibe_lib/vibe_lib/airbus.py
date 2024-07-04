import json
import os
import time
from datetime import datetime
from enum import auto
from typing import Any, Dict, List, Sequence, Tuple
from zipfile import ZipFile

import requests
from fastapi_utils.enums import StrEnum
from shapely import geometry as shpg
from shapely.geometry.base import BaseGeometry

from vibe_core.file_downloader import download_file

from .geometry import wgs_to_utm

DEFAULT_DELAY = 60
DEFAULT_TIMEOUT = 1200
IMAGE_FORMAT = "image/jp2"
LIVING_ATLAS_PROCESSING_LEVEL = "SENSOR"
PRODUCT_TYPE = "pansharpened"
RADIOMETRIC_PROCESSING = "DISPLAY"


class Constellation(StrEnum):
    SPOT = auto()
    PHR = auto()
    PNEO = auto()


class GeometryRelation(StrEnum):
    intersects = auto()
    contains = auto()


class OrderStatus(StrEnum):
    ordered = auto()
    delivered = auto()


class AirBusAPI:
    authentication_url: str = (
        "https://authenticate.foundation.api.oneatlas.airbus.com/"
        "auth/realms/IDP/protocol/openid-connect/token"
    )
    search_url: str = "https://search.foundation.api.oneatlas.airbus.com/api/v2/opensearch"
    price_url: str = "https://data.api.oneatlas.airbus.com/api/v1/prices"
    order_url: str = "https://data.api.oneatlas.airbus.com/api/v1/orders"
    item_url: str = "https://access.foundation.api.oneatlas.airbus.com/api/v1/items"

    def __init__(
        self,
        api_key: str,
        projected_crs: bool,
        constellations: List[Constellation],
        delay: float = DEFAULT_DELAY,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.api_key = api_key
        self.token = self._authenticate()
        self.projected_crs = projected_crs
        self.constellations = constellations
        self.delay = delay  # in seconds
        self.timeout = timeout

    @staticmethod
    def _get_api_key(api_key_filepath: str) -> str:
        with open(api_key_filepath) as f:
            return f.read().strip()

    def _get(self, url: str, **kwargs: Any) -> Dict[str, Any]:
        response = requests.get(url, **kwargs)
        response.raise_for_status()
        return json.loads(response.text)

    def _post(self, url: str, **kwargs: Any) -> Dict[str, Any]:
        response = requests.post(url, **kwargs)
        response.raise_for_status()
        return json.loads(response.text)

    def _authenticate(self):
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = [
            ("apikey", self.api_key),
            ("grant_type", "api_key"),
            ("client_id", "IDP"),
        ]
        response = self._post(self.authentication_url, headers=headers, data=data)
        return response["access_token"]

    def _get_workspace_id(self) -> str:
        headers = {"Authorization": f"Bearer {self.token}", "Cache-Control": "no-cache"}
        response = self._get("https://data.api.oneatlas.airbus.com/api/v1/me", headers=headers)
        return response["contract"]["workspaceId"]

    def _search(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Cache-Control": "no-cache",
            "Content-Type": "application/json",
        }

        response = self._post(self.search_url, headers=headers, json=payload)
        products = [{**r["properties"], "geometry": r["geometry"]} for r in response["features"]]
        return products

    def query(
        self,
        geometry: BaseGeometry,
        date_range: Tuple[datetime, datetime],
        max_cloud_cover: int,
        my_workspace: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Only get results that contain all the geometry (instead of intersecting)

        constellations
          PNEO  0.3m (Neo Pléiades)
          SPOT  1.5m
          PHR   0.5m (Pléiades)

        Cloud cover values used for filtering are for the whole product
        irrespective of the given geometry 😢
        """

        formatted_date = ",".join(
            [dt.astimezone().isoformat().replace("+00:00", "Z") for dt in date_range]
        )
        payload: Dict[str, str] = {
            "geometry": shpg.mapping(geometry),
            "acquisitionDate": f"[{formatted_date}]",
            "constellation": ",".join(self.constellations),
            "cloudCover": f"[0,{max_cloud_cover:d}]",
            "relation": GeometryRelation.intersects if my_workspace else GeometryRelation.contains,
        }
        if my_workspace:
            payload["workspace"] = self._get_workspace_id()
        else:
            payload["processingLevel"] = LIVING_ATLAS_PROCESSING_LEVEL

        return self._search(payload)

    def query_owned(self, geometry: BaseGeometry, acquisition_id: str) -> List[Dict[str, Any]]:
        """
        Query workspace for owned products that match the reference product
        """
        payload: Dict[str, str] = {
            "acquisitionIdentifier": acquisition_id,
            "geometry": shpg.mapping(geometry),
            "relation": GeometryRelation.intersects,
            "workspace": self._get_workspace_id(),
        }
        return self._search(payload)

    def get_product_by_id(self, product_id: str) -> Dict[str, Any]:
        payload: Dict[str, str] = {"id": product_id}
        return self._search(payload)[0]

    def _get_order_params(
        self, product_ids: Sequence[str], roi: BaseGeometry
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Cache-Control": "no-cache",
            "Content-Type": "application/json",
        }
        epsg_code = wgs_to_utm(roi) if self.projected_crs else "4326"
        payload = {
            "kind": "order.data.gb.product",
            "products": [
                {
                    "crsCode": f"urn:ogc:def:crs:EPSG::{epsg_code}",
                    "productType": PRODUCT_TYPE,
                    "radiometricProcessing": RADIOMETRIC_PROCESSING,
                    "aoi": shpg.mapping(roi),
                    "id": pid,
                    "imageFormat": IMAGE_FORMAT,
                }
                for pid in product_ids
            ],
        }
        return headers, payload

    def get_price(self, product_ids: Sequence[str], roi: BaseGeometry) -> Dict[str, Any]:
        headers, payload = self._get_order_params(product_ids, roi)

        response = self._post(self.price_url, headers=headers, json=payload)
        return response

    def place_order(self, product_ids: Sequence[str], roi: BaseGeometry) -> Dict[str, Any]:
        headers, payload = self._get_order_params(product_ids, roi)

        response = self._post(self.order_url, headers=headers, json=payload)
        return response

    def get_order_by_id(self, order_id: str) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.token}"}
        return self._get(f"{self.order_url}/{order_id}", headers=headers)

    def block_until_order_delivered(self, order_id: str) -> Dict[str, Any]:
        start = time.time()
        order = self.get_order_by_id(order_id)
        while order["status"] != OrderStatus.delivered:
            try:
                OrderStatus(order["status"])
            except ValueError:
                raise ValueError(
                    f"Received unexpected status {order['status']} from order {order_id}"
                )
            waiting_time = time.time() - start
            if waiting_time > self.timeout:
                raise RuntimeError(
                    f"Timed out after {waiting_time:.1f}s waiting for order {order_id}"
                )
            time.sleep(self.delay)
            order = self.get_order_by_id(order_id)
        return order

    def download_product(self, product_id: Sequence[str], out_dir: str) -> str:
        headers = {"Authorization": f"Bearer {self.token}"}

        download_url = f"{self.item_url}/{product_id}/download"
        zip_path = os.path.join(out_dir, f"{product_id}.zip")

        download_file(download_url, zip_path, headers=headers)
        with ZipFile(zip_path) as zf:
            zip_member = [f for f in zf.filelist if f.filename.endswith(".JP2")][0]
            # Trick to extract file without the whole directory tree
            # https://stackoverflow.com/questions/4917284/
            zip_member.filename = os.path.basename(zip_member.filename)
            filepath = zf.extract(zip_member, path=out_dir)

        return filepath
