# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from setuptools import find_packages, setup

setup(
    name="vibe_server",
    version="0.0.1",
    author="Microsoft",
    author_email="terravibes@microsoft.com",
    description="TerraVibes Geospatial Platform Package - server package.",
    license="Proprietary",
    keywords="terravibes geospatial",
    packages=find_packages(exclude=["tests*"]),
    python_requires="~=3.8",
    install_requires=[
        "vibe-core",
        "vibe-common",
        "httpx~=0.24.1",
        "fastapi_utils~=0.2.1",
        "grpcio~=1.53.0",
        "dapr==1.13.0",
        "dapr-ext-grpc~=1.12.0",
        "cloudevents~=1.2",
        "fastapi~=0.109.1",
        "fastapi-versioning~=0.10.0",
        "requests~=2.32.0",
        "starlette~=0.36.2",
        "uvicorn~=0.13.4",
        "urllib3~=1.26.8",
        "psutil~=5.9.0",
    ],
    entry_points={
        "console_scripts": [
            "vibe-orchestrator = vibe_server.orchestrator:main_sync",
            "vibe-server = vibe_server.server:main_sync",
            "vibe-sniffer = vibe_server.sniffer:main",
        ]
    },
)
