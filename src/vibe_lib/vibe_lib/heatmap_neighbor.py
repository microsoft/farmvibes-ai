from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
import skgstat as skg
from geopandas import GeoDataFrame
from skgstat import OrdinaryKriging
from sklearn.neighbors import NearestNeighbors


def run_cluster_overlap(
    attribute_name: str,
    reduced_samples: GeoDataFrame,
    minimum_sample_polygons: GeoDataFrame,
    geo_locations: GeoDataFrame,
) -> GeoDataFrame:
    # perform spatial join between minimum sample locations and polygons
    df_overlap = gpd.sjoin(reduced_samples, minimum_sample_polygons)
    df_overlap.rename(
        columns={
            "index_right": "index_overlap",
            "geometry": "geometry_overlap",
        },
        inplace=True,
    )
    df_overlap = df_overlap[["index_overlap", f"{attribute_name}", "geometry_overlap"]]
    # perform spatial join between geolocation points and minimum sample polygons
    geo_locations = gpd.sjoin(geo_locations, minimum_sample_polygons)
    geo_locations.rename(
        columns={
            "index_right": "index_geo_locations",
        },
        inplace=True,
    )
    # assign nutrient values to geolocation points
    out = pd.merge(
        df_overlap,
        geo_locations,
        how="right",
        left_on="index_overlap",
        right_on="index_geo_locations",
    )
    out = out[~out.isna().any(axis=1)]
    out = GeoDataFrame(out[[attribute_name, "geometry"]], geometry="geometry", crs=4326)  # type: ignore
    return out


def run_nearest_neighbor(
    attribute_name: str,
    reduced_samples: GeoDataFrame,
    geo_locations: GeoDataFrame,
) -> GeoDataFrame:
    # preprocess data

    x_ = np.array([reduced_samples.geometry.x, reduced_samples.geometry.y]).T
    y_ = reduced_samples[attribute_name].values
    reduced_samples.drop(columns=["geometry"], inplace=True)
    # train nearest neighbor model
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(x_, y=y_)
    # inference nearest neighbor
    locations = np.array([geo_locations.geometry.x, geo_locations.geometry.y]).T
    _, geo_locations["index_nearest"] = neigh.kneighbors(locations)
    # assign nutrient values to geolocation points
    geo_locations = cast(
        GeoDataFrame,
        geo_locations.merge(reduced_samples, left_on="index_nearest", right_index=True),
    )
    geo_locations = cast(GeoDataFrame, geo_locations[[attribute_name, "geometry"]])
    return geo_locations


def run_kriging_model(
    attribute_name: str,
    reduced_samples: GeoDataFrame,
    geo_locations: GeoDataFrame,
) -> GeoDataFrame:
    # preprocess data
    x_ = np.array([reduced_samples.geometry.x, reduced_samples.geometry.y]).T
    y_ = reduced_samples[attribute_name].values
    # train Variogram using gaussian model
    V = skg.Variogram(x_, y_, model="gaussian", fit_method="trf")
    # train Ordinary Kriging model
    ok = OrdinaryKriging(V, min_points=1, max_points=2, mode="exact")
    # inference Ordinary Krigging
    out_k = ok.transform(geo_locations.geometry.x, geo_locations.geometry.y)
    geo_locations[attribute_name] = out_k
    return geo_locations
