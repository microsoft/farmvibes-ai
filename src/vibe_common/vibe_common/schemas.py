# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from copy import deepcopy
from dataclasses import field, fields
from hashlib import sha256
from itertools import chain
from typing import Any, ClassVar, Dict, List, Optional, Union, cast

import yaml
from pydantic.dataclasses import dataclass
from pystac.item import Item
from typing_extensions import TypedDict  # Required to avoid pydantic error

from vibe_core.data.core_types import BaseVibe, TypeDictVibe, TypeParser
from vibe_core.datamodel import TaskDescription
from vibe_core.utils import rename_keys

from .constants import CONTROL_PUBSUB_TOPIC

ItemDict = Dict[str, Union[Item, List[Item]]]
CacheIdDict = Dict[str, Union[str, List[str]]]
OpDependencies = Dict[str, List[str]]
OpResolvedDependencies = Dict[str, Dict[str, Any]]


class EntryPointDict(TypedDict):
    file: str
    callback_builder: str


@dataclass
class OperationSpec:
    name: str
    root_folder: str
    inputs_spec: TypeDictVibe
    output_spec: TypeDictVibe
    entrypoint: EntryPointDict
    description: TaskDescription
    dependencies: OpDependencies = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    default_parameters: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    image_name: str = CONTROL_PUBSUB_TOPIC

    def __hash__(self):
        return hash(self.name)


def update_parameters(parameters: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict):
            parameters[k] = update_parameters(parameters.get(k, {}), cast(Dict[str, Any], v))
        else:
            if k not in parameters:
                raise ValueError(f"Tried to overwrite non-existent parameter {k}.")
            parameters[k] = v
    return parameters


class OperationParser:
    required_fields: List[str] = "name inputs output parameters entrypoint".split()
    default_version: str = "1.0"

    @classmethod
    def parse(
        cls,
        op_definition_path: str,
        parameters_override: Optional[Dict[str, Any]] = None,
    ) -> OperationSpec:
        op_config = cls._load_config(op_definition_path)
        op_root_folder = os.path.dirname(op_definition_path)

        p = op_config.get("parameters", {})
        default_params: Dict[str, Any] = {} if p is None else p

        inputs = cls._parse_iospec(op_config["inputs"])
        output = cls._parse_iospec(op_config["output"])
        dependencies: OpDependencies = op_config.get("dependencies", {})
        version: str = op_config.get("version", cls.default_version)
        version = str(version) if version is not None else version

        params = deepcopy(default_params)
        if parameters_override is not None:
            params = update_parameters(params, parameters_override)

        description = op_config.get("description", {})
        description = {} if description is None else description
        description = rename_keys(description, {"output": "outputs"})
        description = TaskDescription(**description)

        return OperationSpec(
            name=op_config["name"],
            inputs_spec=inputs,
            output_spec=output,
            entrypoint=EntryPointDict(
                file=op_config["entrypoint"]["file"],
                callback_builder=op_config["entrypoint"]["callback_builder"],
            ),
            parameters=params,
            default_parameters=default_params,
            root_folder=op_root_folder,
            dependencies=dependencies if dependencies is not None else {},
            version=version if version is not None else cls.default_version,
            description=description,
        )

    @classmethod
    def _parse_iospec(cls, iospec: Dict[str, str]) -> TypeDictVibe:
        return TypeDictVibe({k: TypeParser.parse(v) for k, v in iospec.items()})

    @staticmethod
    def _load_config(path: str):
        with open(path, "r") as stream:
            data = yaml.safe_load(stream)

        for opfield in OperationParser.required_fields:
            if opfield not in data:
                raise ValueError(f"Operation config {path} is missing required field {opfield}")

        return data


@dataclass(frozen=True)
class OpRunId:
    name: str
    hash: str


class OpRunIdDict(TypedDict):
    name: str
    hash: str


@dataclass(init=False)
class CacheInfo:
    name: str
    version: str
    hash: str = field(init=False)
    parameters: OpResolvedDependencies = field(init=False)
    ids: Dict[str, Union[str, List[str]]] = field(init=False)

    FIELD_TO_STORAGE: ClassVar[Dict[str, str]] = {
        "version": "vibe_op_version",
        "name": "vibe_op_name",
        "hash": "vibe_op_hash",
        "ids": "vibe_source_items",
        "parameters": "vibe_op_parameters",
    }

    def __init__(
        self,
        name: str,
        version: str = "1.0",
        sources: Optional[ItemDict] = None,
        parameters: OpResolvedDependencies = {},
        **kwargs: Dict[str, Any],
    ):
        self.name = name
        self.version = version.split(".")[0]

        if sources is not None:
            kwargs["sources"] = sources
        kwargs["parameters"] = self.parameters = parameters

        if "ids" not in kwargs:
            if "sources" not in kwargs:
                raise ValueError("CacheInfo missing both `ids` and `sources` fields.")
            self.ids = self._populate_ids(cast(ItemDict, kwargs["sources"]))
        else:
            self.ids = kwargs["ids"]

        if "hash" in kwargs:
            self.hash = cast(str, kwargs["hash"])
        else:
            if "parameters" not in kwargs:
                raise ValueError("CacheInfo missing required parameter `parameters`")
            self.hash = sha256(
                "".join(
                    [
                        self._join_mapping(self.ids),
                        self._join_mapping(cast(OpResolvedDependencies, kwargs["parameters"])),
                        self.version,
                    ]
                ).encode()
            ).hexdigest()

    def as_storage_dict(self):
        return {
            self.FIELD_TO_STORAGE[f.name]: getattr(self, f.name)
            for f in fields(self)  # type: ignore
        }

    @classmethod
    def _compute_or_extract_id(
        cls, thing: Union[Item, BaseVibe, List[Item], List[BaseVibe]]
    ) -> Union[List[str], str]:
        if isinstance(thing, list):
            return [cast(str, cls._compute_or_extract_id(e)) for e in thing]
        return thing.hash_id if hasattr(thing, "hash_id") else thing.id  # type: ignore

    @classmethod
    def _join(cls, thing: Union[Any, List[Any]]) -> str:
        # TODO: this join might lead to collisions, but we're keeping it for now
        # to avoid breaking existing caches
        return "".join([str(i) for i in thing]) if isinstance(thing, list) else str(thing)

    @classmethod
    def _join_mapping(cls, mapping: Union[CacheIdDict, OpResolvedDependencies]) -> str:
        return "".join(
            chain.from_iterable(
                [
                    (k, cls._join_mapping(v) if isinstance(v, dict) else cls._join(v))
                    for k, v in sorted(mapping.items(), key=lambda e: e[0])
                ]
            )
        )

    @classmethod
    def _populate_ids(cls, inputs: ItemDict) -> CacheIdDict:
        return {
            k: cast(List[str], sorted([cls._compute_or_extract_id(e) for e in v]))
            if isinstance(v, list)
            else cls._compute_or_extract_id(v)
            for k, v in inputs.items()
        }
