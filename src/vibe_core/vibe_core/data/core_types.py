"""Core data classes, functions, and constants of FarmVibes.AI."""

import hashlib
import logging
import re
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
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
from .json_converter import dump_to_json

FARMVIBES_AI_BASE_SCHEMA = "schema"
FARMVIBES_AI_BASE_PYDANTIC_MODEL = "pydantic_model"
LOGGER = logging.getLogger(__name__)


BBox = Tuple[float, float, float, float]
"""Type alias for a bounding box, as a tuple of four floats (minx, miny, maxx, maxy)."""

Point = Tuple[float, float]
"""Type alias for a point, as a tuple of two floats (x, y)."""

TimeRange = Tuple[datetime, datetime]
"""Type alias for a time range, as a tuple of two `datetime` objects (start, end)."""


ChipWindow = Tuple[float, float, float, float]
"""Type alias representing a raster chip window, as (col_offset, row_offset, width, height)."""


def gen_guid():
    """Generate a random UUID as a string.

    Returns:
        A random UUID as a string.
    """
    return str(uuid.uuid4())


def gen_hash_id(
    name: str,
    geometry: Union[BaseGeometry, Dict[str, Any]],
    time_range: Tuple[datetime, datetime],
):
    """Generate a hash ID based on a name, a geometry, and a time range.

    Args:
        name: The name associated with the hash ID.
        geometry: The geometry associated with the hash ID,
            either as a `BaseGeometry` object or as a dictionary.
        time_range: The time range associated with the hash ID,
            as a tuple of two `datetime` objects (start, end).

    Returns:
        A hash ID as a hexadecimal string.
    """
    return hashlib.sha256(
        (
            name
            + wkt.dumps(shpg.shape(geometry))
            + time_range[0].isoformat()
            + time_range[1].isoformat()
        ).encode()
    ).hexdigest()


BaseVibeDict = Dict[str, Union["BaseVibe", List["BaseVibe"]]]
BaseUnion = Union["BaseVibe", List["BaseVibe"]]
DataVibeType = Union[Type["BaseVibe"], Type[List["BaseVibe"]]]

InnerIOType = Union[List[Dict[str, Any]], Dict[str, Any]]
OpIOType = Dict[str, InnerIOType]


class TypeDictVibe(Dict[str, DataVibeType]):
    """A dictionary subclass used for type validation in FarmVibes.AI."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> "BaseVibe":
        """Validate a dictionary of values against FarmVibes.AI types.

        This method takes a dictionary of values as input and returns a :class:`BaseVibe` object.
        It validates each value in the dictionary against FarmVibes.AI types using the
        :class:`TypeParser` class. If a value is not a FarmVibes.AI type, a `ValueError` is raised.

        Args:
            v: A dictionary of values to validate.

        Returns:
            A :class:`BaseVibe` object.

        Raises:
            ValueError: If a value in the dictionary is not a FarmVibes.AI type.
        """
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
    """Provide a method for parsing type specifications in FarmVibes.AI.

    It is used to parse the type specifications of ports in :class:`BaseVibe` subclasses.
    """

    logger: logging.Logger = logging.getLogger(f"{__name__}.TypeParser")
    """A logger for the class."""

    type_pattern: "re.Pattern[str]" = re.compile(r"((\w+)\[)?(\w+)\]?")
    """A regular expression pattern to parse type specifications."""

    inherit_pattern = re.compile(r"\s*\@INHERIT\((.*)\)\s*")
    """A regular expression pattern to parse type specifications that inherit from other ports."""

    supported_container_types: List[str] = ["List"]
    """A list of supported container types."""

    container_group: int = 1
    """The group in the regular expression pattern that matches thecontainer type."""

    type_group: int = 2
    """The group in the regular expression pattern that matches the type."""

    @classmethod
    def parse(cls, typespec: str) -> DataVibeType:
        """Parse a type specification string.

        It first checks if the type specification string includes inheritance, and if so,
        returns an :class:`UnresolvedDataVibe` object. Otherwise, it extracts the container and
        data IDs from the type specification string and retrieves the corresponding
        :class:`BaseVibe` subclass from the `data_registry`. If the container or data ID is not
        supported, a `ValueError` is raised.

        Args:
            typespec: A string representing the type specification.

        Returns:
            A :class:`BaseVibe` or a List[:class:`BaseVibe`] object.

        Raises:
            ValueError: If the container ID is not supported or the data ID is not
                a :class:`BaseVibe` subclass.
            KeyError: If the data ID is not found in the `data_registry`.
        """
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
    """Represent an asset in FarmVibes.AI.

    Args:
        id: A string representing the ID of the asset.
        reference: A string representing the path or URL of the asset.
        type: An optional string representing the MIME type of the asset.
    """

    id: str
    """A string representing the ID of the asset."""

    path_or_url: str
    """A string representing the path or URL of the asset."""

    type: Optional[str]
    """An optional string representing the MIME type of the asset."""

    _is_local: bool
    _local_path: Optional[str]

    def __init__(self, reference: str, type: Optional[str], id: str) -> None:
        """Instantiate an AssetVibe object."""
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
        """Return the local path of the asset.

        If the asset is local, this method returns the local path of the asset. If the asset
        is remote, it downloads the asset to a temporary directory (if not previously downloaded)
        and returns the local path of the downloaded file.

        Returns:
            The local path of the asset.
        """
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
        """Return the URL of the asset.

        If the asset is local, this method returns the absolute URI of the local path.
        Otherwise, it returns the original path or URL of the asset.

        Returns:
            The URL of the asset.
        """
        if self._is_local:
            return Path(self.local_path).absolute().as_uri()
        return self.path_or_url


@dataclass
class BaseVibe:
    """Represent a base class for FarmVibes.AI types."""

    schema: ClassVar[Callable[[], Dict[str, Any]]]
    pydantic_model: ClassVar[Callable[[], ModelMetaclass]]

    def __init__(self):
        """Instantiate a new BaseVibe."""
        pass

    def __post_init__(self):
        if "id" not in [f.name for f in fields(self.__class__)]:
            self.id = self.hash_id

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseVibe":
        """Create a :class:`BaseVibe` object from a dictionary of values.

        This method takes a dictionary of values as input and returns a :class:`BaseVibe` object.
        If the class schema includes a bounding box (`bbox`) property, this method calculates the
        bounding box from the `geometry` property using the `shapely.geometry` library.
        If the `geometry` property is missing, a `ValueError` is raised.
        Otherwise, this method creates a new instance of the Pydantic model and returns it.

        Args:
            data: A dictionary of values to create the :class:`BaseVibe` object from.

        Returns:
            A :class:`BaseVibe` object.

        Raises:
            ValueError: If the `geometry` property is missing and the class schema includes
                a `bbox` property.
        """
        if "bbox" in cls.schema()["properties"]:
            try:
                data["bbox"] = shpg.shape(data["geometry"]).bounds
            except KeyError as e:
                raise ValueError(f"Geometry is missing from {data}") from e
        return cls.pydantic_model()(**data)

    @property
    def hash_id(self) -> str:
        """Return the hash ID of the object.

        If the class has an `id` attribute that is a non-empty string, this method returns it.
        Otherwise, it calculates the SHA-256 hash of the JSON representation of the object
        and returns the hexadecimal digest.

        Returns:
            The hash ID of the object.
        """
        if (
            hasattr(self.__class__, "id")
            and isinstance(self.__class__.id, str)  # type: ignore
            and self.id  # type: ignore
        ):
            return self.id  # type: ignore
        return hashlib.sha256(dump_to_json(self).encode()).hexdigest()

    def __init_subclass__(cls, **kwargs):  # type: ignore
        super().__init_subclass__(**kwargs)

        @classmethod
        def schema(cls, *args, **kwargs):  # type: ignore
            return cls.pydantic_model().schema(*args, **kwargs)

        @classmethod
        def pydantic_model(cls):  # type: ignore
            if is_dataclass(cls):
                if issubclass(cls, DataVibe):
                    cls = deepcopy(cls)
                    if "asset_geometry" in cls.__dataclass_fields__:  # type: ignore
                        f = cls.__dataclass_fields__["asset_geometry"]
                        f.type = Dict[str, Any]  # type: ignore

                    @pydataclass
                    class PydanticAssetVibe(AssetVibe):
                        pass

                    @dataclass
                    class Tmp(cls):  # type: ignore
                        assets: List[PydanticAssetVibe]
                        if (
                            hasattr(cls, "__annotations__")
                            and "asset_geometry" in cls.__annotations__
                        ):
                            asset_geometry: Dict[str, Any] = field(default_factory=dict)

                    Model = pydataclass(Tmp)
                    Model.__name__ = cls.__name__  # Model in the repr would confuse users
                    return Model.__pydantic_model__  # type: ignore

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


class UnresolvedDataVibe(Type[BaseVibe], BaseVibe):  # type: ignore
    """Meta type that is equivalent to Python's `type` built-in.

    The output of this class is a new *type*, not a regular object. This is used
    internally by FarmVibes.AI and, in general, should never be instantiated.
    In fact, even if this is instantiated, there's nothing useful that could be
    done with an instance of this (which, again, is a new Type).
    """


def get_filtered_init_field_names(obj: Any, filter_fun: Callable[[Any], bool]):
    """Return a list of filtered field names for an object's `__init__` method.

    Args:
        obj: The object to retrieve the field names from.
        filter_fun: A function that takes a field name as input and returns a boolean indicating
            whether the field should be included in the output list.

    Returns:
        A list of filtered field names for the object's `__init__` method.
    """
    src_fields = get_init_field_names(obj)
    return list(filter(filter_fun, src_fields))


def get_filtered_init_fields(obj: Any, filter_fun: Callable[[Any], bool]):
    """Return a dictionary of filtered fields for an object's `__init__` method.

    Args:
        obj: The object to retrieve the field values from.
        filter_fun: A function that takes a field name as input and returns a boolean indicating
            whether the field should be included in the output dictionary.

    Returns:
        A dictionary of filtered field names and values for the object's `__init__` method.
    """
    field_names = get_filtered_init_field_names(obj, filter_fun)
    obj_dict = asdict(obj)
    return {f: obj_dict[f] for f in field_names}


# TODO consider if we should consolidate geometry and datetime types.
@dataclass
class DataVibe(BaseVibe):
    """Represent a data object in FarmVibes.AI."""

    id: str
    """A string representing the unique identifier of the data object."""

    time_range: TimeRange
    """A :const:`TimeRange` representing the timestamps of to the beginning and end of sample."""

    bbox: BBox = field(init=False)
    """A :const:`BBox` representing the bounding box of the data object.
    This field is calculated from the `geometry` property using the `shapely.geometry` library.
    """

    geometry: Dict[str, Any]
    """A dictionary representing the geometry of the data object."""

    assets: List[AssetVibe]
    """A list of :class:`AssetVibe` objects of the assets associated with the data object."""

    SKIP_FIELDS: ClassVar[Tuple[str, ...]] = ("id", "assets", "hash_id", "bbox")
    """A tuple containing the fields to skip when calculating the hash ID of the object."""

    def __post_init__(self):
        self.bbox = shpg.shape(self.geometry).bounds  # type: ignore
        self.time_range = (
            self.time_range[0].astimezone(timezone.utc),
            self.time_range[1].astimezone(timezone.utc),
        )
        super().__post_init__()

    # Type hint with class that we are defining? https://stackoverflow.com/a/35617812
    @classmethod
    def clone_from(cls, src: "DataVibe", id: str, assets: List[AssetVibe], **kwargs: Any):
        """Create a new :class:`DataVibe` object with updated fields.

        This method takes a source :class:`DataVibe` object, a new `id` string, a list of new
        :class:`AssetVibe` objects, and any additional keyword arguments to update the
        fields of the source object. It returns a new :class:`DataVibe` object with the
        updated fields.

        Args:
            cls: The class of the new :class:`DataVibe` object.
            src: The source :class:`DataVibe` object to clone.
            id: The new `id` string for the cloned object.
            assets: The new list of :class:`AssetVibe` objects for the cloned object.
            kwargs: Additional keyword arguments to update the fields of the cloned object.

        Returns:
            A new :class:`DataVibe` object with the updated fields.
        """
        valid_names = [f for f in get_init_field_names(cls) if f not in cls.SKIP_FIELDS]
        copy_args = get_filtered_init_fields(src, lambda x: x in valid_names)
        copy_args.update(kwargs)
        return cls(id=id, assets=assets, **copy_args)


def get_init_field_names(obj: Type[BaseVibe]) -> List[str]:
    """Return a list of field names for an object's `__init__` method.

    Args:
        obj: The :class:`BaseVibe` class to retrieve the field names from.

    Returns:
        A list of field names for the class's `__init__` method.
    """
    return [f.name for f in fields(obj) if f.init]


@dataclass
class TimeSeries(DataVibe):
    """Represent a time series data object in FarmVibes.AI."""

    pass


@dataclass
class RasterPixelCount(DataVibe):
    """Represent a data object in FarmVibes.AI that stores the pixel count of a raster."""

    pass


@dataclass
class DataSummaryStatistics(DataVibe):
    """Represent a data summary statistics object in FarmVibes.AI."""

    pass


@dataclass
class OrdinalTrendTest(DataVibe):
    """Represent a trend test (Chochan-Armitage) result object in FarmVibes.AI."""

    p_value: float
    """The p-value of the trend test."""
    z_score: float
    """The z-score of the trend test."""


@dataclass
class DataSequence(DataVibe):
    """Represent a sequence of data assets in FarmVibes.AI."""

    idx: int = field(init=False)
    """Number of data objects in the sequence."""

    asset_order: Dict[str, int] = field(default_factory=dict)
    """A dictionary mapping asset IDs to their order in the sequence."""

    asset_time_range: Dict[str, TimeRange] = field(default_factory=dict)
    """A dictionary mapping asset IDs to their time range."""

    asset_geometry: Dict[str, BaseGeometry] = field(default_factory=dict)
    """A dictionary mapping asset IDs to their geometry."""

    def __post_init__(self):
        super().__post_init__()
        lens = [len(i) for i in (self.asset_order, self.asset_time_range, self.asset_geometry)]
        self.idx = lens[0]
        if not all(i == self.idx for i in lens):
            raise ValueError(f"Expected all asset maps to have the same length, found {lens}")

    def add_item(self, item: DataVibe):
        """Add an item to the sequence.

        Args:
            item: The item to be added to the sequence.
        """
        asset = item.assets[0]
        self.add_asset(asset, item.time_range, shpg.shape(item.geometry))

    def add_asset(self, asset: AssetVibe, time_range: TimeRange, geometry: BaseGeometry):
        """Add an asset to the sequence.

        Args:
            asset: The asset to add to the sequence.
            time_range: The time range of the asset.
            geometry: The geometry of the asset.
        """
        self.assets.append(asset)
        self.asset_order[asset.id] = self.idx
        self.asset_time_range[asset.id] = time_range
        self.asset_geometry[asset.id] = geometry
        self.idx += 1

    def get_ordered_assets(self, order_by: Optional[Dict[str, Any]] = None) -> List[AssetVibe]:
        """Get a list of assets in the sequence, ordered by the provided dictionary.

        Args:
            order_by: A dictionary mapping asset IDs to their order in the sequence.
                If None, the assets will be ordered by their default order in the sequence.

        Returns:
            A list of assets in the sequence, ordered by the provided dictionary.
        """
        if order_by is None:
            order_by = self.asset_order
        return sorted(self.assets, key=lambda x: order_by[x.id])


@dataclass
class ExternalReferenceList(DataVibe):
    """Represent a list of external references in FarmVibes.AI."""

    urls: List[str]
    """A list of URLs."""


@dataclass
class ExternalReference(DataVibe):
    """Represent a single external reference in FarmVibes.AI."""

    url: str
    """The URL representing the external reference."""


@dataclass
class GeometryCollection(DataVibe):
    """Represent a geometry collection in FarmVibes.AI."""

    pass


@dataclass
class FoodVibe(BaseVibe):
    """Represent a food object in FarmVibes.AI."""

    dietary_fiber: float
    """The amount of dietary fiber in grams."""

    magnesium: float
    """The amount of magnesium in milligrams."""

    potassium: float
    """The amount of potassium in milligrams."""

    manganese: float
    """The amount of manganese in milligrams."""

    zinc: float
    """The amount of zinc in milligrams."""

    iron: float
    """The amount of iron in milligrams."""

    copper: float
    """The amount of copper in milligrams."""

    protein: float
    """The amount of protein in grams."""

    trp: float  # Tryptophan content
    """The amount of tryptophan in grams."""

    thr: float  # Threonine content
    """The amount of threonine in grams."""

    ile: float  # Isoleucine content
    """The amount of isoleucine in grams."""

    leu: float  # Leucine content
    """The amount of leucine in grams."""

    lys: float  # Lysine content
    """The amount of lysine in grams."""

    met: float  # Methionine content
    """The amount of methionine in grams."""

    cys: float  # Cysteine content
    """The amount of cysteine in grams."""

    phe: float  # Phenylalanine content
    """The amount of phenylalanine in grams."""

    tyr: float  # Tyrosine content
    """The amount of tyrosine in grams."""

    val: float  # Valine content
    """The amount of valine in grams."""

    arg: float  # Arginine content
    """The amount of arginine in grams."""

    his: float  # Histidine content
    """The amount of histidine in grams."""

    fasta_sequence: List[str]
    """A list with the amino acid sequence of the protein."""

    protein_families: List[str]
    """A list with the protein families associated to the food."""

    food_group: str
    """The food group the food belongs to."""


@dataclass
class FoodFeatures(DataVibe):
    """Represent the features of a food in FarmVibes.AI."""

    pass


@dataclass
class ProteinSequence(DataVibe):
    """Represent a protein sequence in FarmVibes.AI."""

    pass


@dataclass
class CarbonOffsetInfo(DataVibe):
    """Represent carbon offset information."""

    carbon: str
    """The carbon offset."""


@dataclass
class GHGFlux(DataVibe):
    """Represent a greenhouse gas (GHG) flux in FarmVibes.AI."""

    scope: str
    """The scope of the GHG flux."""

    value: float
    """The value of the GHG flux."""

    description: Optional[str]
    """An optional description of the GHG flux."""


@dataclass
class GHGProtocolVibe(DataVibe):
    """Represent the inputs to Green House Gas fluxesworkflows.

    This is a dataclass that has many attributes, due to the nature of the
    calculations proposed by the GHG protocol methodology. Not all attributes are required.
    Below we describe all of them, as well as the units they should be in.
    """

    cultivation_area: float  # hectares
    """The area of the field that is cultivated in hectares."""
    total_yield: float  # tonnes
    """The total yield of the field in tonnes."""
    soil_texture_class: Optional[str]  # sand / clay / silt
    """The texture class of the soil (one of the following: "sand", "clay", or "silt")."""
    soil_clay_content: Optional[float]
    """The clay content of the soil in percentage."""
    practice_adoption_period: Optional[int]
    """The number of years that the practice has been adopted."""
    burn_area: Optional[float]
    """The area of the field that is burned in hectares."""
    soil_management_area: Optional[float]
    """The area of the field that is managed in hectares."""

    # fertilizer application {{{
    # Synthetic fertilizers {{{
    urea_amount: Optional[float] = 0.0  # kg per hectare
    """The amount of urea applied to the field in kilograms per hectare."""
    synthetic_fertilizer_amount: Optional[float] = 0.0  # kg per hectare - not urea
    """The amount of synthetic fertilizer applied to the field in kilograms per hectare."""
    synthetic_fertilizer_nitrogen_ratio: Optional[float] = 0.0  # percentage
    """The nitrogen ratio of the synthetic fertilizer applied to the field in percentage."""
    # }}}

    # Soil correction {{{
    limestone_calcite_amount: Optional[float] = 0.0  # kg per hectare
    """The amount of limestone calcite applied to the field in kilograms per hectare."""
    limestone_dolomite_amount: Optional[float] = 0.0  # kg per hectare
    """The amount of limestone dolomite applied to the field in kilograms per hectare."""
    gypsum_amount: Optional[float] = 0.0  # kg per hectare
    """The amount of gypsum applied to the field in kilograms per hectare."""
    # }}}

    # Organic fertilizers {{{
    organic_compound_amount: Optional[float] = 0.0  # kg per hectare
    """The amount of organic compound applied to the field in kilograms per hectare."""
    manure_amount: Optional[float] = 0.0  # kg per hectare
    """The amount of manure applied to the field in kilograms per hectare."""
    manure_birds_amount: Optional[float] = 0.0  # kg per hectare
    """The amount of manure from birds applied to the field in kilograms per hectare."""
    organic_other_amount: Optional[float] = 0.0  # kg per hectare
    """The amount of other organic fertilizer applied to the field in kilograms per hectare."""

    dry_matter_amount: Optional[float] = 0.0  # kg per hectare / Rice
    """The amount of dry matter applied to the field in kilograms per hectare."""
    is_dry_matter_fermented: Optional[bool] = False  # Yes/No / Rice
    """Whether the dry matter is fermented."""

    vinasse_amount: Optional[float] = 0.0  # m^3 per hectare / Sugarcane
    """The amount of vinasse applied to the field in cubic meters per hectare."""
    filter_cake_amount: Optional[float] = 0.0  # kg per hectare / Sugarcane
    """The amount of filter cake applied to the field in kilograms per hectare."""
    filter_cake_application_area: Optional[float] = 0.0  # hectares / Sugarcane
    """The area of the field that is applied with filter cake in hectares."""
    # }}}

    # Green manure {{{
    green_manure_amount: Optional[float] = 0.0  # kg per hectare
    """The amount of green manure applied to the field in kilograms per hectare."""
    green_manure_grass_amount: Optional[float] = 0.0
    """The amount of green manure grass applied to the field in kilograms per hectare."""
    green_manure_legumes_amount: Optional[float] = 0.0
    """The amount of green manure legumes applied to the field in kilograms per hectare."""
    # }}}
    # }}}

    # Rice cultivation {{{
    soil_preparation: Optional[str] = ""  # early / conventional
    """Whether the soil uses "early" or "conventional" preparation."""
    water_regime: Optional[str] = ""
    """The water regime of the field."""
    # }}}

    # Internal fuel {{{
    diesel_type: Optional[str] = "DIESEL"  # diesel(_b2|_b5|_b6|_b7|_b8|_b9|_b10)
    """The type of diesel used in the field."""
    diesel_amount: Optional[float] = 0.0  # liters
    """The amount of diesel used in mechanical operations in the field in liters per hectare."""

    gasoline_amount: Optional[float] = 0.0  # liters
    """The amount of gasoline used in mechanical operations in the field in liters per hectare."""
    ethanol_amount: Optional[float] = 0.0  # liters
    """The amount of ethanol used in mechanical operations in the field in liters per hectare."""
    # }}}

    # Transport fuel {{{
    transport_diesel_type: Optional[str] = "DIESEL"  # diesel(_b2|_b5|_b6|_b7|_b8|_b9|_b10)
    """The type of diesel used in transporting produce from the farm to the market."""
    transport_diesel_amount: Optional[float] = 0.0  # liters
    """Amount of diesel used in transporting produce from farm to market in liters per hectare."""
    # }}}

    current_land_use: str = "conventional_crops"
    """The current land use of the field (can be one of the following:
        "conventional_crops", "direct_seeding", "sugarcane_with_burning", or
        "sugarcane_without_burning").
    """

    previous_land_use: str = "conventional_crops"
    """The previous land use of the field (can be one of the following:
        "conventional_crops", "direct_seeding", "sugarcane_with_burning",
        "native", "sugarcane_without_burning").
    """

    biome: str = ""
    """The biome of the field (can be one of the following "US_FOREST",
        "BRAZIL_AMAZON_FOREST", "BRAZIL_AMAZON_SAVANNA", "BRAZIL_CERRADO",
        "BRAZIL_PANTANAL", "BRAZIL_CAATINGA", "BRAZIL_MATA_ATLANTICA", or
        "BRAZIL_PAMPA").
    """
