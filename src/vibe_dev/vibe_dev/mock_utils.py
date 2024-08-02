# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict

from pydantic import BaseModel


class Request(BaseModel):
    """Mock Request class for testing purposes."""

    text: str
    """Represents the response of the request."""

    def raise_for_status(self) -> Dict[str, int]:
        """Mock raise_for_status method.

        return: A dictionary with a success code.
        """

        return {"success": 200}
