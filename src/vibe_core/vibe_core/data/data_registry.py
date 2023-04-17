import warnings
from typing import Any, Dict, Type, TypeVar, cast

GenericTypeVibe = TypeVar("GenericTypeVibe", bound=Type[Any])
"""Generic type for data registry."""

__DATA_REGISTRY: Dict[str, Type[Any]] = {}


def register_vibe_datatype(classdef: GenericTypeVibe) -> GenericTypeVibe:
    """Registers a class as a data type in the FarmVibes.AI data registry.

    :param classdef: The class to register.

    :return: The class.
    """
    id = get_name(classdef)
    if id in __DATA_REGISTRY:
        warnings.warn(f"Class {id} already registered.", DeprecationWarning, stacklevel=2)
        return cast(GenericTypeVibe, retrieve(id))
    __DATA_REGISTRY[id] = classdef

    return cast(GenericTypeVibe, classdef)


def retrieve(id: str) -> Type[Any]:
    """
    Retrieves a registered data type from the FarmVibes.AI data registry.

    :param id: The ID of the data type to retrieve.

    :return: The registered data type.
    """
    return __DATA_REGISTRY[id]


def get_name(classdef: Type[Any]) -> str:
    """
    Gets the name of a class.

    :param classdef: The class to get the name of.

    :return: The name of the class.
    """
    return classdef.__name__


def get_id(classdef: Type[Any]) -> str:
    """
    Gets the ID of a class.

    :param classdef: The class to get the ID of.

    :return: The ID of the class.
    """
    id = get_name(classdef)

    return id
