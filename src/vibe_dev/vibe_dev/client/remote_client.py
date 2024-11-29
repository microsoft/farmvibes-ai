# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from vibe_core.client import FarmvibesAiClient

PPE_URL = "https://ppe-terravibes-api.57fb76945e6d4b66a912.eastus.aksapp.io/"


def get_ppe_vibe_client(url: str = PPE_URL):
    return FarmvibesAiClient(url)
