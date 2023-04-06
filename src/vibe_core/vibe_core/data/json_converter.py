import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any

from pydantic.dataclasses import dataclass as pydataclass
from pydantic.main import BaseModel


class DataclassJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any):
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
    return json.dumps(
        data,
        allow_nan=False,
        cls=DataclassJSONEncoder,
        **kwargs,
    )
