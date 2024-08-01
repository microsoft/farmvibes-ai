# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from setuptools import find_packages, setup

setup(
    name="vibe_notebook",
    version="0.0.1",
    author="Microsoft",
    author_email="terravibes@microsoft.com",
    packages=find_packages(),
    description="Shared notebook library for FarmVibes.AI notebooks.",
    install_requires=[
        "numpy",
        "geopandas",
        "pandas",
        "shapely",
        "rasterio",
        "vibe_core",
    ],
)
