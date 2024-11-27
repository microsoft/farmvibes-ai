# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from setuptools import find_packages, setup

setup(
    name="vibe_dev",
    version="0.0.1",
    author="Microsoft",
    author_email="terravibes@microsoft.com",
    description="TerraVibes Geospatial Platform Package - vibe dev.",
    license="Proprietary",
    keywords="terravibes geospatial",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "pytest",
        "pytest-azurepipelines",
        "pytest-cov",
        "onnx~=1.16.0",
    ],
    python_requires="~=3.8",
    entry_points={
        "console_scripts": [
            "vibe-local-run = vibe_dev.localrunner:main",
        ]
    },
    package_data={
        "vibe_dev": [
            "testing/fake_ops/fake/*.py",
            "testing/fake_ops/fake/*.yaml",
            "testing/fake_workflows/*.yaml",
        ]
    },
)
