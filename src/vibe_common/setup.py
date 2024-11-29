# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from setuptools import find_packages, setup

setup(
    name="vibe-common",
    version="0.0.1",
    author="Microsoft",
    author_email="terravibes@microsoft.com",
    description="TerraVibes Geospatial Platform Package - vibe common package.",
    license="Proprietary",
    keywords="terravibes geospatial",
    packages=find_packages(exclude=["tests*"]),
    python_requires="~=3.8",
    install_requires=[
        "aiohttp~=3.9.0",
        "aiohttp-retry~=2.8.3",
        "azure-keyvault>=4.1.0",
        "jsonschema~=4.6",
        "requests~=2.32.0",
        "cloudevents~=1.2",
        "grpcio~=1.53.0",
        "dapr~=1.13.0",
        "fastapi_utils~=0.2.1",
        "pyyaml~=6.0.1",
        "vibe_core",
        "debugpy",
        "azure-identity~=1.14.0",
        "azure-storage-blob>=12.5.0",
        "uvicorn~=0.13.4",
        "uvloop~=0.17.0",
        "fastapi~=0.109.1",
        "httptools~=0.6.0",
        "gunicorn~=21.2.0",
        "opentelemetry-api~=1.20.0",
        "opentelemetry-sdk~=1.20.0",
        "opentelemetry-exporter-otlp~=1.20.0",
        "opentelemetry-instrumentation~=0.41b0",
    ],
)
