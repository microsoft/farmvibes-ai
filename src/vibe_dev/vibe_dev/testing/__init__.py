# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"
