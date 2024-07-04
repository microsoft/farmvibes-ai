from setuptools import find_packages, setup

setup(
    name="vibe_lib",
    version="0.0.1",
    author="Microsoft",
    author_email="terravibes@microsoft.com",
    description="TerraVibes Geospatial Platform Package - vibe lib.",
    license="Proprietary",
    keywords="terravibes geospatial",
    packages=find_packages(exclude=["tests*"]),
    python_requires="~=3.8",
    install_requires=["numpy", "geopandas", "rasterio~=1.2"],
)
