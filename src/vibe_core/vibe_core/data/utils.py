"""Utilities for interacting with STAC items and serialization/deserialization."""

import json
from dataclasses import fields
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Tuple,
    Type,
    TypeVar,
    Union,
    _type_repr,  # type: ignore
    cast,
    get_args,
    get_origin,
    overload,
)

from pydantic import BaseModel
from pystac.asset import Asset
from pystac.item import Item
from shapely import geometry as shpg
from shapely.geometry.base import BaseGeometry

from . import data_registry
from .core_types import (
    AssetVibe,
    BaseVibe,
    DataVibe,
    DataVibeType,
    get_filtered_init_fields,
    get_init_field_names,
)

T = TypeVar("T", bound=BaseVibe, covariant=True)
V = TypeVar("V")


class FieldConverter(NamedTuple):
    """A named tuple representing a field converter."""

    serializer: Callable[[Any], Any]
    """A function that serialize a value."""

    deserializer: Callable[[Any], Any]
    """A function that deserialize a value."""


def is_json_serializable(x: Any) -> bool:
    """Check if a field is JSON serializable by Python's default serializer.

    Args:
        x: The value to check.

    Returns:
        True if the value is JSON serializable, False otherwise.
    """
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def to_isoformat(x: datetime) -> str:
    """Convert a datetime object to an ISO format string.

    Args:
        x: The datetime object to convert.

    Returns:
        The ISO format string.
    """
    return x.isoformat()


class StacConverter:
    """Convert :class:`BaseVibe` objects to STAC Items."""

    field_converters = {
        BaseGeometry: FieldConverter(shpg.mapping, shpg.shape),
        datetime: FieldConverter(to_isoformat, datetime.fromisoformat),
    }
    """A dictionary mapping field types to field converters."""

    VIBE_DATA_TYPE_FIELD = "terravibes_data_type"
    """The name of the field that contains the data type of the :class:`BaseVibe` object."""

    BASEVIBE_FALLBACK_DATETIME = datetime(1970, 1, 1)
    """The fallback datetime to use for :class:`BaseVibe` objects."""

    def __init__(self):
        """Instantiate a StacConverter object."""
        pass

    def sanitize_properties(self, properties: Dict[Any, Any]) -> Dict[Any, Any]:
        """Sanitize a dictionary of properties to ensure they are JSON serializable.

        Args:
            properties: The dictionary of properties to sanitize.

        Returns:
            The sanitized dictionary of properties.
        """
        out = {}
        for k, v in properties.items():
            if is_json_serializable(v):
                out[k] = v
            else:
                Warning(f"Field {k} is not JSON serializable, it will not be added to STAC item.")

        return out

    def _serialize_type(self, field_value: Any, field_type: Any) -> Any:
        converter = self.field_converters.get(field_type)
        if converter is None:
            if isinstance(field_value, BaseModel):
                # We have to do this, otherwise our sanitizer will filter out
                # this value
                return json.loads(field_value.json())
            return field_value
        return converter.serializer(field_value)

    def _deserialize_type(self, field_value: Any, field_type: Any) -> Any:
        converter = self.field_converters.get(field_type)
        if converter is None:
            return field_value
        return converter.deserializer(field_value)

    def convert_field(
        self, field_value: Any, field_type: Any, converter: Callable[[Any, Any], Any]
    ) -> Any:
        """Convert a field value to a given type, using a converter function.

        Args:
            field_value: The value of the field to convert.
            field_type: The type to convert the field value to.
            converter: The converter function to use.

        Returns:
            The converted field value.
        """
        t_origin = get_origin(field_type)
        t_args = get_args(field_type)
        if t_origin is list and len(t_args) == 1:
            return [self.convert_field(f, t_args[0], converter) for f in field_value]
        if t_origin is dict and t_args:
            return {k: self.convert_field(v, t_args[1], converter) for k, v in field_value.items()}
        if t_origin is tuple and t_args:
            if len(t_args) == 2 and t_args[1] == ...:
                return tuple(self.convert_field(f, t_args[0], converter) for f in field_value)
            return tuple(
                self.convert_field(f, ta, converter) if ta is datetime else f
                for f, ta in zip(field_value, t_args)
            )
        return converter(field_value, field_type)

    def serialize_fields(
        self, field_values: Dict[str, Any], field_types: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Serialize a dictionary of fields.

        Args:
            field_values: The dictionary of field values to serialize.
            field_types: The dictionary of field types to serialize to.

        Returns:
            The serialized dictionary of field values.
        """
        return {
            k: self.convert_field(v, field_types[k], self._serialize_type)
            for k, v in field_values.items()
        }

    def deserialize_fields(
        self, field_values: Dict[str, Any], field_types: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deserialize a dictionary of fields.

        Args:
            field_values: The dictionary of field values to deserialize.
            field_types: The dictionary of field types to deserialize to.

        Returns:
            The deserialized dictionary of field values.
        """
        return {
            k: self.convert_field(v, field_types[k], self._deserialize_type)
            for k, v in field_values.items()
        }

    @overload
    def to_stac_item(self, input: BaseVibe) -> Item: ...

    @overload
    def to_stac_item(self, input: List[BaseVibe]) -> List[Item]: ...

    def to_stac_item(self, input: Union[List[BaseVibe], BaseVibe]):
        """Convert :class`BaseVibe` to STAC.

        Convert a :class`BaseVibe` or a list of :class`BaseVibe` to a STAC item
        or a list of STAC items.

        Args:
            input: The :class`BaseVibe` or list of :class`BaseVibe` to convert.

        Returns:
            A STAC item or a list of STAC items.
        """
        if isinstance(input, list):
            return [
                self._to_stac_impl(i) if isinstance(i, DataVibe) else self._base_vibe_to_stac(i)
                for i in input
            ]

        if isinstance(input, DataVibe):
            return self._to_stac_impl(input)
        return self._base_vibe_to_stac(input)

    def _extract_properties(self, input: BaseVibe) -> Dict[str, Any]:
        # If this object inherits from BaseVibe but not from DataVibe, then the
        # base is BaseVibe. Otherwise, the base is DataVibe.
        # Whatever the base is, it is the input to `get_init_field_names`
        regular_fields = get_init_field_names(
            BaseVibe if not isinstance(input, DataVibe) else DataVibe
        )
        properties = get_filtered_init_fields(input, lambda x: x not in regular_fields)
        property_types = {f.name: f.type for f in fields(input) if f.name in properties}
        properties = self.serialize_fields(properties, property_types)
        return properties

    def _base_vibe_to_stac(self, input: BaseVibe) -> Item:
        properties = self._extract_properties(input)
        properties = self.sanitize_properties(properties)

        extra_fields = {self.VIBE_DATA_TYPE_FIELD: data_registry.get_id(type(input))}

        item = Item(
            id=input.id,
            datetime=(
                self.BASEVIBE_FALLBACK_DATETIME
                if not hasattr(input, "datetime")
                else input.datetime  # type: ignore
            ),
            bbox=None,
            geometry=None,
            properties=properties,
            extra_fields=extra_fields,
        )
        return item

    def _to_stac_impl(self, input: DataVibe) -> Item:
        properties = self._extract_properties(input)

        properties["start_datetime"] = input.time_range[0].isoformat()
        properties["end_datetime"] = input.time_range[1].isoformat()

        extra_fields = {self.VIBE_DATA_TYPE_FIELD: data_registry.get_id(type(input))}

        properties = self.sanitize_properties(properties)

        item = Item(
            id=input.id,
            datetime=input.time_range[0],
            bbox=list(input.bbox),
            geometry=input.geometry,
            properties=properties,
            extra_fields=extra_fields,
        )

        for asset in input.assets:
            item.add_asset(
                key=asset.id,
                asset=Asset(href=asset.path_or_url, media_type=asset.type),
            )

        return item

    @overload
    def from_stac_item(self, input: Item) -> BaseVibe: ...

    @overload
    def from_stac_item(self, input: List[Item]) -> List[BaseVibe]: ...

    def from_stac_item(self, input: Union[Item, List[Item]]) -> Union[BaseVibe, List[BaseVibe]]:
        """Convert STAC to :class`BaseVibe`.

        Convert a STAC item or a list of STAC items to a :class`BaseVibe`
        or a list of :class`BaseVibe`.

        Args:
            input: The STAC item or list of STAC items to convert.

        Returns:
            A :class`BaseVibe` or a list of :class`BaseVibe`.
        """
        if isinstance(input, list):
            return [self._from_stac_impl(i) for i in input]

        return self._from_stac_impl(input)

    def _from_stac_impl(self, input: Item) -> BaseVibe:
        # Figuring out type to create
        vibe_data_type = self.resolve_type(input)
        # Need to find the necessary arguments to the constructor of the type
        init_fields = list(get_init_field_names(vibe_data_type))
        init_field_types = {f.name: f.type for f in fields(vibe_data_type) if f.name in init_fields}
        # Read properties from item stac into init fields
        in_props: Dict[str, Any] = input.properties  # type: ignore
        data_kw = {f: in_props[f] for f in init_fields if f in in_props}
        data_kw = self.deserialize_fields(data_kw, init_field_types)
        data_kw.update(self._build_extra_kwargs(input, vibe_data_type))

        # Creating actual object
        return vibe_data_type(**data_kw)

    def _build_extra_kwargs(self, input: Item, type: Type[BaseVibe]) -> Dict[str, Any]:
        # Adding DataVibe-specific fields - think of better mechanism to do this...
        data_kw = {}

        if issubclass(type, DataVibe):
            data_kw["id"] = input.id
            data_kw["time_range"] = convert_time_range(input)
            data_kw["geometry"] = input.geometry  # type: ignore
            data_kw["assets"] = [
                AssetVibe(reference=a.href, type=a.media_type, id=id)
                for id, a in input.assets.items()
            ]

        return data_kw

    def resolve_type(self, input: Item) -> Type[BaseVibe]:
        """Resolve the type of a :class`BaseVibe` object from a STAC item.

        Args:
            input: The STAC item to resolve the type from.

        Returns:
            The type of :class`BaseVibe`.
        """
        extra_fields: Dict[str, Any] = input.extra_fields  # type: ignore
        if self.VIBE_DATA_TYPE_FIELD not in extra_fields:
            return BaseVibe

        return cast(
            Type[BaseVibe],
            data_registry.retrieve(extra_fields[self.VIBE_DATA_TYPE_FIELD]),
        )


def convert_time_range(item: Item) -> Tuple[datetime, datetime]:
    """Convert the time range of a STAC item to a tuple of datetimes.

    Args:
        item: The STAC item to convert the time range for.

    Returns:
        A tuple of datetimes representing the start and end of the time range.
    """
    conv_foo = datetime.fromisoformat
    props: Dict[str, Any] = item.properties  # type: ignore
    if "start_datetime" in props and "end_datetime" in props:
        return (
            conv_foo(props["start_datetime"]),
            conv_foo(props["end_datetime"]),
        )

    assert item.datetime is not None

    return (item.datetime, item.datetime)


@overload
def serialize_stac(arg: Item) -> Dict[str, Any]: ...


@overload
def serialize_stac(arg: List[Item]) -> List[Dict[str, Any]]: ...


def serialize_stac(arg: Union[Item, List[Item]]):
    """Serialize a STAC item or a list of STAC items to a dictionary or a list of dictionaries.

    Args:
        arg: The STAC item or list of STAC items to serialize.

    Returns:
        A dictionary or a list of dictionaries representing the STAC item
        or list of STAC items.
    """
    if isinstance(arg, list):
        return [item.to_dict(include_self_link=False) for item in arg]

    return arg.to_dict(include_self_link=False)


@overload
def deserialize_stac(arg: Dict[str, Any]) -> Item: ...


@overload
def deserialize_stac(arg: List[Dict[str, Any]]) -> List[Item]: ...


def deserialize_stac(arg: Union[List[Dict[str, Any]], Dict[str, Any]]):
    """Deserialize dict to STAC.

    Deserialize a dictionary or a list of dictionaries to a STAC item
    or a list of STAC items.

    Args:
        arg: The dictionary or list of dictionaries to deserialize.

    Returns:
        A STAC item or a list of STAC items.
    """
    item_builder = Item.from_dict

    if isinstance(arg, list):
        return [item_builder(in_dict) for in_dict in arg]

    return item_builder(arg)


@overload
def serialize_input(input_data: BaseVibe) -> Dict[str, Any]: ...


@overload
def serialize_input(input_data: List[T]) -> List[Dict[str, Any]]: ...


@overload
def serialize_input(
    input_data: Dict[str, Union[T, List[T]]],
) -> Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]]: ...


def serialize_input(input_data: Any) -> Any:
    """Serialize :class`BaseVibe` to STAC.

    Serialize a single :class`BaseVibe` object, or a list or dictionary of them,
    to a STAC item or a list or dictionary of STAC items.

    Args:
        input_data: The :class`BaseVibe` object or a list or dictionary
            of :class`BaseVibe` objects to serialize.

    Returns:
        A list, a dictionary or a single STAC Item representing the :class`BaseVibe`
        object.

    Raises:
        NotImplementedError: If the input data is not a :class`BaseVibe` object,
            or a list or dictionary of :class`BaseVibe`.
    """
    # Dictionary where keys are workflow sources
    if isinstance(input_data, dict):
        return {k: serialize_input(v) for k, v in input_data.items()}
    # Input is a list of elements
    if isinstance(input_data, list):
        return [serialize_input(i) for i in input_data]
    if isinstance(input_data, BaseVibe):
        return serialize_stac(StacConverter().to_stac_item(input_data))
    raise NotImplementedError(f"Unable to serialize {input_data.__class__} objects to JSON")


def get_base_type(vibetype: DataVibeType) -> Type[BaseVibe]:
    """Determine the base type of a typing specification.

    Args:
        vibetype: The type to determine the base type of.

    Returns:
        The base type of vibetype.

    Raises:
        ValueError: If the type hierarchy contains nested container types
            (e.g., List[List[:class:`DataVibe`]]).

    Doctests:
    >>> get_base_type(DataVibe)
    vibe_core.data.DataVibe
    >>> get_base_type([List[DataVibe])
    vibe_core.data.DataVibe
    """
    if not (is_container_type(vibetype) or isinstance(vibetype, type)):
        raise ValueError(f"Argument {vibetype} is not a type")
    if isinstance(vibetype, type):
        return cast(Type[T], vibetype)
    levels = 1
    tmp = get_args(vibetype)
    while tmp is not None and is_container_type(tmp[0]):
        origin = get_origin(tmp[0])
        if origin is None:
            raise AssertionError("Found a None type in the hierarchy")
        if not issubclass(origin, list):
            raise ValueError(f"Container type {origin.__name__} is not supported")
        tmp = get_args(tmp[0])
        levels += 1
    if levels > 1:
        raise ValueError("Nested container types are not supported")
    return tmp[0]


def is_container_type(typeclass: Union[Type[V], List[Type[V]]]) -> bool:
    """Check if a type is a container type.

    Args:
        typeclass: The type to check.

    Returns:
        True if the type is a container type, False otherwise.
    """
    return bool(get_args(typeclass))


def is_vibe_list(typeclass: DataVibeType) -> bool:
    """Check if a type is a list of :class`BaseVibe` objects.

    Args:
        typeclass: The type to check.

    Returns:
        True if the type is a list of :class`BaseVibe` objects, False otherwise.
    """
    origin = get_origin(typeclass)
    return origin is not None and issubclass(origin, list)


def get_most_specific_type(types: List[DataVibeType]) -> DataVibeType:
    """Determine the most specific type of a list of types.

    Args:
        types: The list of types to determine the most specific type of.

    Returns:
        The most specific type of types.

    Raises:
        ValueError: If the types are not compatible.
    """
    t_set = set(get_base_type(t) for t in types)
    for t in t_set:
        if all(issubclass(t, tt) for tt in t_set):
            break
    else:
        types_str = ", ".join([f"'{_type_repr(t)}'" for t in t_set])
        raise ValueError(f"Types {types_str} are not compatible")
    if all(is_container_type(tt) for tt in types):
        return List[t]
    return t
