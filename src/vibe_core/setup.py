from setuptools import find_packages, setup

setup(
    name="vibe-core",
    version="2023.02.24",
    author="Microsoft",
    author_email="eywa-devs@microsoft.com",
    description="FarmVibes.AI Geospatial Platform Package - vibe core package.",
    license="Proprietary",
    keywords="farmvibes-ai geospatial",
    packages=find_packages(exclude=["tests*"]),
    python_requires="~=3.8",
    install_requires=[
        "jsonschema~=4.6",
        "pydantic~=1.8.2",
        "strenum~=0.4.7",
        "shapely>=1.7.1",
        "requests>=2.27",
        "pystac~=1.6.0",
        "hydra-zen~=0.7",
        "rich~=12.5.1",
    ],
)
