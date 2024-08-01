# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from vibe_core.client import get_default_vibe_client

from .remote_client import get_ppe_vibe_client

__all__ = ["get_default_vibe_client", "get_ppe_vibe_client"]
