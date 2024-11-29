"""JSON serialization/deserialization utilities."""

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any

from pydantic.dataclasses import dataclass as pydataclass
from pydantic.main import BaseModel


class DataclassJSONEncoder(json.JSONEncoder):
    """Extend `json.JSONEncoder` to support encoding dataclasses and pydantic models."""

    def default(self, obj: Any):
        """Encode a dataclass or pydantic model to JSON.

        Args:
            obj: The object to encode.

        Returns:
            The JSON representation of the object.
        """
        if is_dataclass(obj):
            cls = pydataclass(obj.__class__).__pydantic_model__
            exclude = {"hash_id"} if hasattr(obj.__class__, "hash_id") else {}
            return json.loads(cls(**asdict(obj)).json(allow_nan=False, exclude=exclude))
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, BaseModel):
            return json.loads(obj.json(allow_nan=False))
        return super().default(obj)


def dump_to_json(data: Any, **kwargs: Any) -> str:
    """Serialize an object to JSON using :class:`DataclassJSONEncoder`.

    Args:
        data: The object to serialize to JSON.
        **kwargs: Additional keyword arguments to pass to the `json.dumps` method.

    Returns:
        A JSON string representation of the object.
    """
    return json.dumps(
        data,
        allow_nan=False,
        cls=DataclassJSONEncoder,
        **kwargs,
    )
