import hashlib
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    get_args,
    get_origin,
)

from pydantic.dataclasses import dataclass as pydataclass
from pydantic.main import BaseModel, ModelMetaclass
from shapely import geometry as shpg
from shapely import wkt
from shapely.geometry.base import BaseGeometry

from ..file_downloader import build_file_path, download_file
from ..uri import is_local
from . import data_registry

FARMVIBES_AI_BASE_SCHEMA = "schema"
FARMVIBES_AI_BASE_PYDANTIC_MODEL = "pydantic_model"
LOGGER = logging.getLogger(__name__)


BBox = Tuple[float, float, float, float]
TimeRange = Tuple[datetime, datetime]


def gen_guid():
    return str(uuid.uuid4())


def gen_hash_id(
    name: str, geometry: Union[BaseGeometry, Dict[str, Any]], time_range: Tuple[datetime, datetime]
):
    return hashlib.sha256(
        (
            name
            + wkt.dumps(shpg.shape(geometry))
            + time_range[0].isoformat()
            + time_range[1].isoformat()
        ).encode()
    ).hexdigest()


DataVibeDict = Dict[str, Union["DataVibe", List["DataVibe"]]]
BaseUnion = Union["BaseVibe", List["BaseVibe"]]
DataVibeType = Union[Type["BaseVibe"], Type[List["BaseVibe"]]]

InnerIOType = Union[List[Dict[str, Any]], Dict[str, Any]]
OpIOType = Dict[str, InnerIOType]


class TypeDictVibe(Dict[str, DataVibeType]):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> "BaseVibe":
        try:
            for key in v:
                if isinstance(v[key], str):
                    v[key] = TypeParser.parse(v[key])
                elif isinstance(get_origin(v[key]), Type):
                    args = get_args(v[key])[0]
                    origin = get_origin(args)
                    if origin is not None and issubclass(origin, List):
                        base = get_args(args)[0]
                        if issubclass(base, BaseVibe):
                            continue
                    elif issubclass(args, BaseVibe):
                        continue
                    else:
                        raise ValueError(f"Value for key {key} is not a FarmVibes.AI type")
            return v
        except TypeError:
            raise ValueError


class TypeParser:
    logger: logging.Logger = logging.getLogger(f"{__name__}.TypeParser")
    type_pattern: "re.Pattern[str]" = re.compile(r"((\w+)\[)?(\w+)\]?")
    inherit_pattern = re.compile(r"\s*\@INHERIT\((.*)\)\s*")
    supported_container_types: List[str] = ["List"]
    container_group: int = 1
    type_group: int = 2

    @classmethod
    def parse(cls, typespec: str) -> DataVibeType:
        inherit = cls.inherit_pattern.findall(typespec)
        if inherit:
            # What `parse` returns needs to be a Type, and not a class instance. Because of
            # that, we have to game the type system here, such that we output something that
            # is a valid DataVibeType, and that has the name of the port we are inheriting from.
            # So, by instantiating `UnresolvedDataVibe()`, which itself inherits from type,
            # we are creating a new Type[BaseVibe]. The line below, by the way, is equivalent
            # to `type(inherit[0][1], (), {})`, with the additional `Type[BaseVibe]` parent.
            inherit_type = UnresolvedDataVibe(inherit[0], (), {})
            return cast(DataVibeType, inherit_type)

        typespec = typespec.replace("typing.", "").replace("vibe_core.data.", "")
        matches = cls.type_pattern.findall(typespec)
        containerid = matches[0][cls.container_group]
        dataid = matches[0][cls.type_group]

        if containerid and containerid not in cls.supported_container_types:
            raise ValueError(f"Operation uses unsupported container {containerid}")

        try:
            datavibe = data_registry.retrieve(dataid)
            if not issubclass(datavibe, BaseVibe):
                raise ValueError(
                    f"Operation uses unsupported type {data_registry.get_name(datavibe)}"
                )
            datavibe_list = List[datavibe]  # type: ignore
            return datavibe_list if containerid else datavibe
        except KeyError:
            raise KeyError(f"Unable to find type {dataid}")


@dataclass
class AssetVibe:
    type: Optional[str]
    id: str
    path_or_url: str
    _is_local: bool
    _local_path: Optional[str]

    def __init__(self, reference: str, type: Optional[str], id: str) -> None:
        self._is_local = is_local(reference)
        self._local_path = reference if self._is_local else None
        self.path_or_url = reference
        self.type = type
        self.id = id
        self._tmp_dir = TemporaryDirectory()
        if type is None:
            LOGGER.warning(f"Asset {self} created without defined mimetype")

    def __del__(self):
        try:
            self._tmp_dir.cleanup()
        except (AttributeError, FileNotFoundError):
            LOGGER.info(f"Unable to clean temporary directory related to VibeAsset {self.url}")

    @property
    def local_path(self) -> str:
        if self._is_local:
            return cast(str, self._local_path)
        # This is a remote asset
        if self._local_path:
            # The download was previously done
            return self._local_path
        # The asset is remote and there is no previous download
        file_path = build_file_path(
            self._tmp_dir.name, gen_guid(), "" if self.type is None else self.type
        )
        self._local_path = download_file(self.url, file_path)
        return self._local_path

    @property
    def url(self) -> str:
        if self._is_local:
            return Path(self.local_path).absolute().as_uri()
        return self.path_or_url


class BaseVibe:
    schema: Callable[[], Dict[str, Any]]
    pydantic_model: Callable[[], ModelMetaclass]

    def __init__(self):
        pass

    def __init_subclass__(cls, **kwargs):  # type: ignore
        super().__init_subclass__(**kwargs)

        @classmethod
        def schema(cls, *args, **kwargs):  # type: ignore
            return cls.pydantic_model().schema(*args, **kwargs)

        @classmethod
        def pydantic_model(cls):  # type: ignore
            if is_dataclass(cls):
                if issubclass(cls, DataVibe):

                    @pydataclass
                    class PydanticAssetVibe(AssetVibe):
                        pass

                    @pydataclass
                    class Tmp(cls):
                        assets: List[PydanticAssetVibe]

                        class Config:
                            underscore_attrs_are_private = True
                            arbitrary_types_allowed = True

                    return Tmp.__pydantic_model__

                return pydataclass(cls).__pydantic_model__
            if issubclass(cls, BaseModel):
                return cls
            raise NotImplementedError(f"{cls.__name__} is not a dataclass")  # type:ignore

        if not hasattr(cls, FARMVIBES_AI_BASE_SCHEMA):
            setattr(cls, FARMVIBES_AI_BASE_SCHEMA, schema)

        if not hasattr(cls, FARMVIBES_AI_BASE_PYDANTIC_MODEL):
            setattr(cls, FARMVIBES_AI_BASE_PYDANTIC_MODEL, pydantic_model)

        try:
            data_registry.retrieve(cls.__name__)
        except KeyError:
            data_registry.register_vibe_datatype(cls)


class UnresolvedDataVibe(Type[BaseVibe], BaseVibe):
    """Meta type that is equivalent to Python's `type` built-in.

    The output of this class is a new *type*, not a regular object. This is used
    internally by FarmVibes.AI and, in general, should never be instantiated.
    In fact, even if this is instantiated, there's nothing useful that could be
    done with an instance of this (which, again, is a new Type).
    """


def get_filtered_init_field_names(obj: Any, filter_fun: Callable[[Any], bool]):
    src_fields = get_init_field_names(obj)
    return list(filter(filter_fun, src_fields))


def get_filtered_init_fields(obj: Any, filter_fun: Callable[[Any], bool]):
    field_names = [f for f in get_filtered_init_field_names(obj, filter_fun)]
    obj_dict = asdict(obj)
    return {f: obj_dict[f] for f in field_names}


# TODO consider if we should consolidate geometry and datetime types.
@dataclass
class DataVibe(BaseVibe):
    id: str
    time_range: TimeRange  # Timestamps corresponding to the beginning and end of sample
    bbox: BBox = field(init=False)
    geometry: Dict[str, Any]  # This should be the
    assets: List[AssetVibe]

    def __post_init__(self):
        self.bbox = shpg.shape(self.geometry).bounds  # type: ignore
        self.time_range = (
            self.time_range[0].astimezone(timezone.utc),
            self.time_range[1].astimezone(timezone.utc),
        )

    # Type hint with class that we are defining? https://stackoverflow.com/a/35617812
    @classmethod
    def clone_from(cls, src: "DataVibe", id: str, assets: List[AssetVibe], **kwargs: Any):
        valid_names = [f for f in get_init_field_names(cls) if f not in ("id", "assets")]
        copy_args = get_filtered_init_fields(src, lambda x: x in valid_names)
        copy_args.update(kwargs)
        return cls(id=id, assets=assets, **copy_args)


def get_init_field_names(obj: Type[DataVibe]) -> Generator[str, None, None]:
    return (f.name for f in fields(obj) if f.init)


@dataclass
class TimeSeries(DataVibe):
    pass


@dataclass
class DataSummaryStatistics(DataVibe):
    pass


@dataclass
class DataSequence(DataVibe):
    asset_order: Dict[str, int] = field(default_factory=dict)
    asset_time_range: Dict[str, TimeRange] = field(default_factory=dict)
    asset_geometry: Dict[str, BaseGeometry] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        lens = [len(i) for i in (self.asset_order, self.asset_time_range, self.asset_geometry)]
        self.idx = lens[0]
        if not all(i == self.idx for i in lens):
            raise ValueError(f"Expected all asset maps to have the same length, found {lens}")

    def add_item(self, item: DataVibe):
        asset = item.assets[0]
        self.add_asset(asset, item.time_range, shpg.shape(item.geometry))

    def add_asset(self, asset: AssetVibe, time_range: TimeRange, geometry: BaseGeometry):
        self.assets.append(asset)
        self.asset_order[asset.id] = self.idx
        self.asset_time_range[asset.id] = time_range
        self.asset_geometry[asset.id] = geometry
        self.idx += 1

    def get_ordered_assets(self, order_by: Optional[Dict[str, Any]] = None) -> List[AssetVibe]:
        if order_by is None:
            order_by = self.asset_order
        return sorted(self.assets, key=lambda x: order_by[x.id])


@dataclass
class ExternalReferenceList(DataVibe):
    urls: List[str]


@dataclass
class ExternalReference(DataVibe):
    url: str


@dataclass
class GeometryCollection(DataVibe):
    pass
