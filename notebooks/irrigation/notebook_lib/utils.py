from datetime import datetime, timedelta

import geopandas as gpd
import pystac_client
from pystac.item import Item
from shapely import geometry as shpg


# Define a function to process buffer geometry
def process_buffer_geometry(geom, ds):
    # Convert the input geometry to the CRS of the dataset
    proj_geom = gpd.GeoSeries(geom, crs="epsg:4326").to_crs(ds.rio.crs).iloc[0]

    # Clip the dataset using the projected geometry
    ds_crop = ds.rio.clip([proj_geom])

    # Extract the mask (values) of the cropped dataset
    mask = ds_crop.values[0]

    # Return the resulting mask
    return mask


def query_catalog(geometry, time_range, query):
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    datetime = (
        "/".join(i.strftime("%Y-%m-%d") for i in time_range) if time_range is not None else None
    )
    search = catalog.search(
        collections=["landsat-c2-l2"],
        intersects=shpg.mapping(geometry) if geometry is not None else None,
        datetime=datetime,
        query=query,
    )
    return list(search.get_items())


# Function to process cloud criteria
def cloud_criteria(item: Item) -> bool:
    max_cloud_cover = 15  # Assuming you have this value defined somewhere
    return item.properties["landsat:cloud_cover_land"] < max_cloud_cover


# Function to filter out date based on the intersection
def filter_items_by_intersection(ok_items, geom, min_intersection_area):
    filtered_items = []
    for item in ok_items:
        item_polygon = shpg.shape(item.geometry)
        intersection_area = geom.intersection(item_polygon).area
        if intersection_area >= min_intersection_area:
            filtered_items.append(item)
    return filtered_items


def select_target_time_given_cloud_cover(
    latitude: float,
    longitude: float,
    year: int,
    month: int,
    day: int,
    buffer_radius: float,
    max_cloud_cover: int,
):
    lat = latitude
    lon = longitude
    target_time = datetime(year, month, day)

    geom = shpg.Point(lon, lat).buffer(buffer_radius, cap_style=3)

    delta = timedelta(days=15)

    # Minimum intersection area required (in square degrees)
    min_intersection_area = 1.0 * (geom.area)

    while True:
        cur_time_range = (target_time - delta, target_time + delta)

        ls_items = query_catalog(
            geometry=geom,
            time_range=cur_time_range,
            query={
                "platform": {"in": ["landsat-8"]},
            },
        )

        ok_items = [i for i in ls_items if cloud_criteria(i)]

        filtered_items = filter_items_by_intersection(ok_items, geom, min_intersection_area)

        if not filtered_items:
            delta *= 2
            continue

        chosen_item = sorted(
            filtered_items, key=lambda x: abs(target_time.astimezone() - x.datetime)
        )[0]
        break

    # Print the chosen image information
    target_time = chosen_item.datetime
    formatted_date = target_time.strftime("%Y-%m-%d")
    print("Entered longitude and latitude is:", lon, ",", lat)
    print("Assigned Search Date is:", year, "-", month, "-", day)
    print(
        f"Chosen Image Date with Equal or Less than {max_cloud_cover}% Cloud Condition in the Area is:",
        formatted_date,
    )

    return geom, target_time
