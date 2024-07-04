import io
import mimetypes
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Sequence, Tuple, cast
from xml.etree.ElementTree import Element, ElementTree

import numpy as np
import planetary_computer as pc
import requests
import rioxarray as rio  # noqa: F401
import xarray as xr
from numpy.typing import NDArray
from pystac.item import Item
from pystac_client import Client
from rasterio.warp import Resampling
from rioxarray.merge import merge_arrays
from shapely import geometry as shpg
from shapely.geometry.base import BaseGeometry

from vibe_core.data import AssetVibe, Raster, gen_guid
from vibe_lib.raster import get_crs

CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
DATE_FORMAT = "%Y-%m-%d"

BBox = Tuple[float, float, float, float]
Angles = Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]


def query_catalog(roi: BBox, time_range: Tuple[datetime, datetime]):
    """
    Query the planetary computer for items that intersect with the desired RoI in the time range
    """
    catalog = Client.open(CATALOG_URL)
    search = catalog.search(
        collections=[COLLECTION],
        bbox=roi,
        datetime="/".join(i.strftime(DATE_FORMAT) for i in time_range),
    )
    items = list(search.get_items())
    return items


def get_catalog_items(raster: Raster, tol: timedelta = timedelta(days=5)) -> List[Item]:
    """
    Get sentinel2 tiles that intersect with the raster geometry
    within a tolerance of the raster datetime
    """
    geom = shpg.shape(raster.geometry)
    roi = cast(BBox, geom.bounds)
    raster_dt = raster.time_range[0]
    time_range = (raster_dt - tol, raster_dt + tol)
    items = query_catalog(roi, time_range)
    # Filter items by closest date
    dates = list(set(cast(datetime, item.datetime) for item in items))
    date_distance = cast(NDArray[Any], [abs(raster_dt - d).total_seconds() for d in dates])
    closest_date = dates[np.argmin(date_distance)]
    items = [item for item in items if item.datetime == closest_date]

    # Return items necessary to cover all the spatial extent of the raster
    return filter_necessary_items(geom, items)


def filter_necessary_items(poly: BaseGeometry, items: Sequence[Item]) -> List[Item]:
    """
    Greedily filter the items so that only a subset necessary to cover all the raster spatial extent
    is returned
    """

    def area_func(item: Item) -> float:
        bbox = item.bbox
        assert bbox is not None
        return -shpg.box(*bbox, ccw=True).intersection(poly).area

    sorted_items = sorted(items, key=area_func)

    # Get item with largest intersection
    item = sorted_items[0]
    assert item
    assert item.bbox is not None
    item_box = shpg.box(*item.bbox, ccw=True)
    if poly.within(item_box):
        return [item]
    return [item] + filter_necessary_items(poly - item_box, sorted_items[1:])


def get_xml_data(item: Item) -> ElementTree:
    """
    Get granule metadata XML from the planetary computer STAC item
    """
    href = item.assets["granule-metadata"].href
    signed_href = pc.sign(href)
    response = requests.get(signed_href)
    return ET.parse(io.BytesIO(response.content))


def parse_grid_params(tree: ElementTree) -> Tuple[float, float, float, float, str]:
    """
    Parse center grid coordinates and grid resolution from the metadata XML
    """
    res = 10
    height, width = [
        int(cast(str, v.text))
        for node in tree.iter("Size")
        if node.attrib["resolution"] == str(res)
        for tag in ("NROWS", "NCOLS")
        for v in node.iter(tag)
    ]
    xmin, ymax = [
        int(cast(str, v.text))
        for node in tree.iter("Geoposition")
        if node.attrib["resolution"] == str(res)
        for tag in ("ULX", "ULY")
        for v in node.iter(tag)
    ]

    xc = xmin + res * width / 2
    yc = ymax - res * height / 2
    res_x = float(cast(str, next(tree.iter("COL_STEP")).text))
    res_y = -float(cast(str, next(tree.iter("ROW_STEP")).text))
    crs = cast(str, next(tree.iter("HORIZONTAL_CS_CODE")).text)
    return xc, yc, res_x, res_y, crs


def parse_angle_grids(node: Element) -> NDArray[Any]:
    """
    Parse zenith and azimuth grids from XML node
    Returns array of shape 2 (zenith, azimuth) x H x W
    """
    angles = (
        np.array(
            [
                [
                    [cast(str, line.text).split(" ") for line in mat.iter("VALUES")]
                    for mat in node.iter(za)
                ]
                for za in ["Zenith", "Azimuth"]
            ]
        )
        .astype(float)
        .squeeze()  # Get rid of the singleton dimension from node.iter(za)
    )
    return angles


def get_view_angles(tree: ElementTree) -> Tuple[NDArray[Any], NDArray[Any]]:
    """
    Parse view angles from XML tree, join per-band detector grids, then average over bands
    """
    grid_list = [
        [
            parse_angle_grids(node)
            for node in tree.iter("Viewing_Incidence_Angles_Grids")
            if node.attrib["bandId"] == str(bi)
        ]
        for bi in range(13)
    ]
    # Band indices x Detector ID x Zenith or Azimuth x H x W
    partial_grids = np.array(grid_list)
    # Join partial grids from all detectors
    n = np.nan_to_num(partial_grids).sum(axis=1)
    d = np.isfinite(partial_grids).sum(axis=1)
    angles = n / d
    # Get the average from all bands
    view_zenith_mean, view_azimuth_mean = angles.mean(axis=0)
    return view_zenith_mean, view_azimuth_mean


def get_sun_angles(tree: ElementTree) -> Tuple[NDArray[Any], NDArray[Any]]:
    """
    Parse sun angles from XML tree
    """
    node = next(tree.iter("Sun_Angles_Grid"))
    sun_zenith, sun_azimuth = parse_angle_grids((node))
    return sun_zenith, sun_azimuth


def to_georeferenced_array(
    angle_grid: NDArray[Any], center: Tuple[float, float], resolution: Tuple[float, float], crs: str
) -> xr.DataArray:
    """"""
    height, width = angle_grid.shape
    grid_x, grid_y = (
        np.linspace(c - (dim - 1) / 2 * res, c + (dim - 1) / 2 * res, dim)
        for c, res, dim in zip(center, resolution, (width, height))
    )

    array = xr.DataArray(angle_grid[None], {"band": [1], "y": grid_y, "x": grid_x})
    array.rio.set_crs(crs)
    return array


def get_angles_from_item(
    item: Item,
) -> Angles:
    """
    Get georeferenced view and sun angle grids by querying planetary computer,
    parsing the metadata XML for grid coordinates and values, and joining per-band view grids.
    Returns mean view zenith, mean view azimuth, sun zenith, and sun azimuth grids, respectively.
    """
    tree = get_xml_data(item)
    xc, yc, res_x, res_y, crs = parse_grid_params(tree)
    angles = (*get_view_angles(tree), *get_sun_angles(tree))
    # get geospatial grid for these arrays
    return cast(
        Angles,
        tuple(
            to_georeferenced_array(angle_grid, (xc, yc), (res_x, res_y), crs)
            for angle_grid in angles
        ),
    )


def get_angles(raster: Raster, tol: timedelta = timedelta(days=5)) -> Angles:
    """
    Fetch view and sun angle grids, according to the raster geometry and time range.
    Time range is assumed to be one value. The closest visit is used in case there is no samples
    for the exact date. In case the geometry spans multiple tiles, the angle grids will be merged.
    Grids are reprojected to native tif CRS and clipped according to the geometry.
    Angle grid resolution is kept at 5000m.
    Returns mean view zenith, mean view azimuth, sun zenith, and sun azimuth grids, respectively.
    """
    geom = shpg.shape(raster.geometry)
    items = get_catalog_items(raster, tol)
    items = filter_necessary_items(geom, items)
    angles_list = zip(*(get_angles_from_item(item) for item in items))

    raster_crs = get_crs(raster)
    return cast(
        Angles,
        tuple(
            merge_arrays(
                [
                    ang.rio.reproject(raster_crs, resampling=Resampling.bilinear, nodata=np.nan)
                    for ang in angles
                ]
            ).rio.clip([geom], crs="epsg:4326", all_touched=True)
            for angles in angles_list
        ),
    )


class CallbackBuilder:
    def __init__(self, tolerance: int):
        self.tmp_dir = TemporaryDirectory()
        self.tolerance = timedelta(days=tolerance)

    def __call__(self):
        def fcover_callback(raster: Raster) -> Dict[str, Raster]:
            angles = xr.concat(get_angles(raster, tol=self.tolerance), dim="band")
            uid = gen_guid()
            out_path = os.path.join(self.tmp_dir.name, f"{uid}.tif")
            angles.rio.to_raster(out_path)
            asset = AssetVibe(reference=out_path, type=mimetypes.types_map[".tif"], id=uid)
            out_raster = Raster.clone_from(
                raster,
                id=gen_guid(),
                assets=[asset],
                bands={
                    k: v
                    for v, k in enumerate(
                        ["view_zenith", "view_azimuth", "sun_zenith", "sun_azimuth"]
                    )
                },
            )
            return {"angles": out_raster}

        return fcover_callback

    def __del__(self):
        self.tmp_dir.cleanup()
