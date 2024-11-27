# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import hashlib
from calendar import monthrange
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import pytz
import rasterio
import requests
from dateutil.parser import isoparse
from dateutil.relativedelta import relativedelta
from pystac import MediaType
from pystac.asset import Asset
from pystac.item import Item
from shapely import geometry as shpg
from shapely.geometry import Polygon, mapping

from vibe_core.data import ChirpsProduct, DataVibe
from vibe_core.data.core_types import BBox


class ChirpsCollection:
    INI = datetime(1981, 1, 1, tzinfo=timezone.utc)  # first day Chirps is available
    VALID_FREQ = {"daily", "monthly"}
    VALID_RES = {"p05", "p25"}

    def __init__(self, freq: str, res: str):
        if freq not in self.VALID_FREQ:
            raise ValueError(
                f"Invalid Chirps frequency {freq} - valid options are {','.join(self.VALID_FREQ)}"
            )
        if res not in self.VALID_RES:
            raise ValueError(
                f"Invalid Chirps resolution {res} - valid options are {','.join(self.VALID_RES)}"
            )
        if freq == "monthly" and res != "p05":
            raise ValueError("Monthly Chirps is only available on p05 resolution")

        self.freq = freq
        self.res = res
        self.end = self.get_latest_chirps()
        # all bbox are the same, so we pick from the latest file
        self.bbox, self.footprint = self.get_bbox_and_footprint(self.end)
        self.var = "precipitation"

    def url(self, year: int) -> str:
        if self.freq == "monthly":
            return "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/cogs/"
        else:
            return (
                f"https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/"
                f"cogs/{self.res}/{year}/"
            )

    def fname(self, date: datetime) -> str:
        if self.freq == "monthly":
            return f"chirps-v2.0.{date.year}.{date.month:02}.cog"
        else:
            return f"chirps-v2.0.{date.year}.{date.month:02}.{date.day:02}.cog"

    def get_latest_chirps(self) -> datetime:
        ini = self.INI
        end = datetime(
            datetime.today().year,
            datetime.today().month,
            datetime.today().day,
            tzinfo=timezone.utc,
        )
        date = end
        for year in range(end.year, ini.year - 1, -1):
            text = requests.get(self.url(year)).text
            while date >= datetime(year, 1, 1, tzinfo=timezone.utc):
                if text.find(self.fname(date)) > 0:
                    return date
                if self.freq == "daily":
                    date -= timedelta(days=1)
                else:
                    date -= relativedelta(months=1)
                    date = date.replace(day=monthrange(date.year, date.month)[1])
            date = datetime(year - 1, 12, 31, tzinfo=timezone.utc)
        raise ValueError("no Chirps file found")  # this point should never be reached

    def get_bbox_and_footprint(self, date: datetime) -> Tuple[BBox, Polygon]:
        url = self.url(date.year) + self.fname(date)
        with rasterio.open(url) as ds:
            bounds = ds.bounds
            bbox = (bounds.left, bounds.bottom, bounds.right, bounds.top)
            footprint = shpg.box(*bounds)
        return (bbox, footprint)

    def get_chirps_list(
        self, time_range: Tuple[datetime, datetime]
    ) -> List[Tuple[datetime, str, str]]:
        tr = [dt.astimezone(pytz.timezone("UTC")) for dt in time_range]
        end_range = (
            tr[1]
            if self.freq == "daily"
            else tr[1].replace(day=monthrange(tr[1].year, tr[1].month)[1])
        )
        if (
            time_range[1].timestamp() < self.INI.timestamp()
            or time_range[0].timestamp() > self.end.timestamp()
        ):
            raise ValueError(
                f"Invalid time range {time_range[0].isoformat()} - "
                f"{time_range[1].isoformat()} - valid values are in the range"
                f"{self.INI.isoformat()} - {self.end.isoformat()}"
            )
        ini = tr[0] if tr[0] >= self.INI else self.INI
        end = end_range if end_range <= self.end else self.end
        date = end
        res = []
        while date >= ini:
            url = self.url(date.year) + self.fname(date)
            fname = self.fname(date)
            res.append((date, url, fname))
            if self.freq == "daily":
                date -= timedelta(days=1)
            else:
                date -= relativedelta(months=1)
                date = date.replace(day=monthrange(date.year, date.month)[1])
        return res

    def _get_id(self, fname: str) -> str:
        return hashlib.sha256(f"{self.res}_{fname}".encode()).hexdigest()

    def query(
        self,
        roi: Optional[BBox] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Item]:
        if roi is not None:
            pgon = shpg.box(*roi)
            if not pgon.intersects(self.footprint):
                return []
        ini = time_range[0] if time_range is not None else self.INI
        end = time_range[1] if time_range is not None else self.end
        chirpsl = self.get_chirps_list((ini, end))
        res = []
        for date, url, fname in chirpsl:
            id = self._get_id(fname)
            if ids is not None and id not in ids:
                continue
            item = self._create_item(date, url, id)
            res.append(item)
        return res

    def _create_item(self, date: datetime, url: str, id: str) -> Item:
        item = Item(
            id=id,
            geometry=mapping(self.footprint),
            bbox=[self.bbox[i] for i in range(4)],
            datetime=date,
            properties={},
        )
        asset = Asset(href=url, media_type=MediaType.COG)
        item.add_asset(self.var, asset)
        return item

    def query_by_id(self, id: Union[str, List[str]]) -> List[Item]:
        if isinstance(id, str):
            ids = [id]
        else:
            ids = id
        res = []
        for date, url, fname in self.get_chirps_list((self.INI, self.end)):
            id = self._get_id(fname)
            if id in ids:
                item = self._create_item(date, url, id)
                res.append(item)
        return res


def convert_product(item: Dict[str, Any], freq: str) -> ChirpsProduct:
    date = isoparse(item["properties"]["datetime"]).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    if freq == "daily":
        time_range = (date, date)
    else:
        time_range = (date.replace(day=1), date)
    url = item["assets"]["precipitation"]["href"]
    output = ChirpsProduct(
        id=item["id"],
        time_range=time_range,
        geometry=item["geometry"],
        assets=[],
        url=url,
    )
    return output


class CallbackBuilder:
    def __init__(self, freq: str, res: str):
        self.freq = freq
        self.res = res

    def __call__(self):
        def list_chirps(
            input_item: DataVibe,
        ) -> Dict[str, List[ChirpsProduct]]:
            collection = ChirpsCollection(self.freq, self.res)
            items = collection.query(roi=input_item.bbox, time_range=input_item.time_range)

            products = [convert_product(item.to_dict(), freq=self.freq) for item in items]

            if not products:
                raise RuntimeError(
                    f"No product found for time range {input_item.time_range} "
                    f"and geometry {input_item.geometry}"
                )
            return {"chirps_products": products}

        return list_chirps
