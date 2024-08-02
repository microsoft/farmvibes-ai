# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from setuptools import find_packages, setup

setup(
    name="vibe_agent",
    version="0.0.1",
    author="Microsoft",
    author_email="terravibes@microsoft.com",
    description="TerraVibes Geospatial Platform Package - vibe package.",
    license="Proprietary",
    keywords="terravibes geospatial",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "aiorwlock~=1.3.0",
        "azure-cosmos~=4.5.0",
        "pystac~=1.6.0",
        "azure-identity~=1.14.0",
        "azure-storage-blob>=12.5.0",
        "httpx~=0.24.1",
        "shapely>=1.7.1",
        "PyYAML~=6.0.1",
        "pebble~=4.6.3",
        "grpcio~=1.53.0",
        "dapr==1.13.0",
        "dapr-ext-grpc~=1.12.0",
        "redis~=4.6.0",
        "hiredis~=2.2.0",
        "vibe-core",
        "vibe-common",
    ],
    entry_points={
        "console_scripts": [
            "vibe-worker = vibe_agent.launch_worker:main",
            "vibe-cache = vibe_agent.launch_cache:main",
            "vibe-data-ops = vibe_agent.launch_data_ops:main",
        ]
    },
)
