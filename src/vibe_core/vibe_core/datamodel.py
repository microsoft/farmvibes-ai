import codecs
import json
import zlib
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import auto
from typing import Any, Dict, Final, List, Optional, Union, cast
from uuid import UUID

from dateutil.parser import parse
from strenum import StrEnum

from .data.core_types import OpIOType
from .data.json_converter import dump_to_json

SUMMARY_DEFAULT_FIELDS: Final[List[str]] = ["id", "workflow", "name", "details.status"]


@dataclass
class Message:
    """Dataclass that represents an API message."""

    message: str
    """The message."""

    id: Optional[str] = None
    """The id of the message."""

    location: Optional[str] = None
    """The location of the message."""


@dataclass
class Region:
    """Dataclass that represents a region."""

    name: str
    """The name of the region."""

    geojson: Dict[str, Any] = field(default_factory=dict)
    """The geojson of the region."""


@dataclass
class SpatioTemporalJson:
    """Dataclass that represents a spatio temporal json."""

    start_date: datetime
    """The start date of the spatio temporal json."""

    end_date: datetime
    """The end date of the spatio temporal json."""

    geojson: Dict[str, Any]
    """The geojson of the spatio temporal json."""

    def __post_init__(self):
        for attr in ("start_date", "end_date"):
            if isinstance(some_date := getattr(self, attr), str):
                setattr(self, attr, parse(some_date))


@dataclass
class RunBase:
    """Base dataclass for a run."""

    name: str
    """The name of the run."""
    workflow: Union[str, Dict[str, Any]]
    """The workflow of the run."""
    parameters: Optional[Dict[str, Any]]
    """The parameters of the run."""

    def __post_init__(self):
        if isinstance(self.workflow, str):
            try:
                self.workflow = json.loads(self.workflow)
            except json.decoder.JSONDecodeError:
                pass


@dataclass
class RunConfigInput(RunBase):
    """Dataclass that represents a run config input."""

    user_input: Union[SpatioTemporalJson, Dict[str, Any], List[Any]]
    """The user input of the run config (usually a region/geometry and time range)."""

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.user_input, dict):
            try:
                self.user_input = SpatioTemporalJson(**self.user_input)
            except TypeError:
                # We need this because of BaseVibe.
                pass


class RunStatus(StrEnum):
    """Enum that represents the status of a run."""

    pending = auto()
    """The run is pending"""
    queued = auto()
    """The run is queued."""
    running = auto()
    """The run is running."""
    failed = auto()
    """The run has failed."""
    done = auto()
    """The run is done."""
    cancelled = auto()
    """The run is cancelled."""
    cancelling = auto()
    """The run is cancelling."""

    @staticmethod
    def finished(status: "RunStatus"):
        """Checks if a run has finished.

        This method checks if a run status is either
        :attr:`vibe_core.datamodel.RunStatus.done`,
        :attr:`vibe_core.datamodel.RunStatus.cancelled`, or
        :attr:`vibe_core.datamodel.RunStatus.failed`.

        :param status: The status to check.

        :return: Whether the run has finished.

        """
        return status in (RunStatus.done, RunStatus.cancelled, RunStatus.failed)


@dataclass
class RunDetails:
    """Dataclass that encapsulates the details of a run."""

    start_time: Optional[datetime] = None
    """The start time of the run."""
    submission_time: Optional[datetime] = None
    """The submission time of the run."""
    end_time: Optional[datetime] = None
    """The end time of the run."""
    reason: Optional[str] = None
    """A description of the reason for the status of the run."""
    status: RunStatus = RunStatus.pending  # type: ignore
    """The status of the run."""
    subtasks: Optional[List[Any]] = None
    """Details about the subtasks of the run."""

    def __post_init__(self):
        for time_field in ("start_time", "submission_time", "end_time"):
            attr = cast(Union[str, datetime, None], getattr(self, time_field))
            if isinstance(attr, str):
                setattr(self, time_field, parse(attr))


@dataclass
class RunConfig(RunConfigInput):
    """Dataclass that represents a run config."""

    id: UUID
    """The id of the run config."""
    details: RunDetails
    """The details of the run config."""
    task_details: Dict[str, RunDetails]
    """The details of the tasks of the run config."""
    spatio_temporal_json: Optional[SpatioTemporalJson]
    """The spatio temporal json of the run config."""
    output: str = ""
    """The output of the run."""

    def set_output(self, value: OpIOType):  # pydantic won't let us use a property setter
        """Sets the output of the run config.

        :param value: The value to set the output to.
        """
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
    """Dataclass that represents a run config for the user."""

    output: OpIOType
    """The output of the run."""

    @classmethod
    def from_runconfig(cls, run_config: RunConfig):
        """Creates a :class:`RunConfigUser` from a :class:`RunConfig`.

        :param run_config: The run config to create the user run config from.

        :return: The user run config.

        """
        rundict = asdict(run_config)
        output = rundict.pop("output")
        rcu = cls(**rundict)
        rcu.output = json.loads(decode(output)) if output else {}
        return rcu

    @staticmethod
    def finished(status: "RunStatus"):
        """Checks if a run has finished.

        This method checks if a given status is either
        :attr:`vibe_core.datamodel.RunStatus.done`,
        :attr:`vibe_core.datamodel.RunStatus.cancelled`, or
        :attr:`vibe_core.datamodel.RunStatus.failed`.

        :param status: The status to check.

        :return: Whether the run has finished.

        """

        return status in (RunStatus.done, RunStatus.cancelled, RunStatus.failed)


@dataclass
class TaskDescription:
    """Dataclass that represents a task description."""

    inputs: Dict[str, str] = field(default_factory=dict)
    """The inputs of the task."""
    outputs: Dict[str, str] = field(default_factory=dict)
    """The outputs of the task."""
    parameters: Dict[str, str] = field(default_factory=dict)
    """The task parameters."""
    task_descriptions: Dict[str, str] = field(default_factory=dict)
    """The descriptions of subtasks."""
    short_description: str = ""
    """The short description of the task."""
    long_description: str = ""
    """The long description of the task."""


def encode(data: str) -> str:
    """Encodes a string using zlib and base64 encoding.

    This function compresses the data string with zlib and then encodes it into a base64 string.

    :param data: The string to be encoded.

    :return: The encoded string.
    """
    return codecs.encode(zlib.compress(data.encode("utf-8")), "base64").decode("utf-8")  # JSON 😞


def decode(data: str) -> str:
    """Decodes the given data using zlib and base64 encodings.

    :param data: The string to decode.

    :return: The decoded string.
    """
    return zlib.decompress(codecs.decode(data.encode("utf-8"), "base64")).decode("utf-8")  # JSON 😞
