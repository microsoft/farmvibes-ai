import codecs
import json
import zlib
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from enum import auto
from typing import Any, Dict, Final, List, Optional, Union
from uuid import UUID

from dateutil.parser import parse
from pydantic import BaseModel
from pydantic.dataclasses import dataclass as pydataclass
from strenum import StrEnum

from .data.core_types import OpIOType

SUMMARY_DEFAULT_FIELDS: Final[List[str]] = ["id", "workflow", "name", "details.status"]


@dataclass
class Message:
    message: str
    id: Optional[str] = None
    location: Optional[str] = None


@dataclass
class Region:
    name: str
    geojson: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpatioTemporalJson:
    start_date: datetime
    end_date: datetime
    geojson: Dict[str, Any]

    def __post_init__(self):
        for attr in ("start_date", "end_date"):
            if isinstance(some_date := getattr(self, attr), str):
                setattr(self, attr, parse(some_date))


@dataclass
class RunBase:
    name: str
    workflow: Union[str, Dict[str, Any]]
    parameters: Optional[Dict[str, Any]]

    def __post_init__(self):
        if isinstance(self.workflow, str):
            try:
                self.workflow = json.loads(self.workflow)
            except json.decoder.JSONDecodeError:
                pass


@dataclass
class RunConfigInput(RunBase):
    user_input: Union[SpatioTemporalJson, Dict[str, Any], List[Any]]

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.user_input, dict):
            try:
                self.user_input = SpatioTemporalJson(**self.user_input)
            except TypeError:
                # We need this because of BaseVibe.
                pass


class RunStatus(StrEnum):
    pending = auto()
    running = auto()
    failed = auto()
    done = auto()
    cancelled = auto()
    cancelling = auto()

    @staticmethod
    def finished(status: "RunStatus"):
        return status in (RunStatus.done, RunStatus.cancelled, RunStatus.failed)


@dataclass
class RunDetails:
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    reason: Optional[str] = None
    status: RunStatus = RunStatus.pending

    def __post_init__(self):
        if isinstance(self.start_time, str):
            self.start_time = parse(self.start_time)
        if isinstance(self.end_time, str):
            self.end_time = parse(self.end_time)


@dataclass
class RunConfig(RunConfigInput):
    id: UUID
    details: RunDetails
    task_details: Dict[str, RunDetails]
    spatio_temporal_json: Optional[SpatioTemporalJson]
    output: str = ""

    def set_output(self, value: OpIOType):  # pydantic won't let us use a property setter
        self.output = encode(dump_to_json(value))

    def __post_init__(self):
        if isinstance(self.details, dict):
            self.details = RunDetails(**self.details)

        if self.spatio_temporal_json is not None and isinstance(self.spatio_temporal_json, dict):
            try:
                self.spatio_temporal_json = SpatioTemporalJson(**self.spatio_temporal_json)
            except TypeError:
                pass

        for k, v in self.task_details.items():
            if isinstance(v, dict):
                self.task_details[k] = RunDetails(**v)

        super().__post_init__()


class RunConfigUser(RunConfig):
    output: OpIOType

    @classmethod
    def from_runconfig(cls, run_config: RunConfig):
        rundict = asdict(run_config)
        output = rundict.pop("output")
        rcu = cls(**rundict)
        rcu.output = json.loads(decode(output)) if output else {}
        return rcu

    @staticmethod
    def finished(status: "RunStatus"):
        return status in (RunStatus.done, RunStatus.cancelled, RunStatus.failed)


@dataclass
class TaskDescription:
    inputs: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, str] = field(default_factory=dict)
    task_descriptions: Dict[str, str] = field(default_factory=dict)
    short_description: str = ""
    long_description: str = ""


class DataclassJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        if is_dataclass(obj):
            cls = pydataclass(obj.__class__).__pydantic_model__
            return json.loads(cls(**asdict(obj)).json(allow_nan=False))
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, BaseModel):
            return json.loads(obj.json(allow_nan=False))
        return super().default(obj)


def encode(data: str) -> str:
    return codecs.encode(zlib.compress(data.encode("utf-8")), "base64").decode("utf-8")  # JSON 😞


def decode(data: str) -> str:
    return zlib.decompress(codecs.decode(data.encode("utf-8"), "base64")).decode("utf-8")  # JSON 😞


def dump_to_json(data: Any, **kwargs: Any) -> str:
    return json.dumps(
        data,
        allow_nan=False,
        cls=DataclassJSONEncoder,
        **kwargs,
    )
