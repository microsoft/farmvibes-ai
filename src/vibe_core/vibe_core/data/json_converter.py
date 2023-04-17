import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any

from pydantic.dataclasses import dataclass as pydataclass
from pydantic.main import BaseModel


class DataclassJSONEncoder(json.JSONEncoder):
    """
    A class that extends the `json.JSONEncoder` class to support
    encoding of dataclasses and pydantic models.
    """

    def default(self, obj: Any):
        """Encodes a dataclass or pydantic model to JSON.

        :param obj: The object to encode.

        :return: The JSON representation of the object.
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
    """Serializes an object to JSON using :class:`DataclassJSONEncoder`.

    :param data: The object to serialize to JSON.

    :param **kwargs: Additional keyword arguments to pass to the `json.dumps` method.

    :return: A JSON string representation of the object.
    """
    return json.dumps(
        data,
        allow_nan=False,
        cls=DataclassJSONEncoder,
        **kwargs,
    )
